from plato.trainers import basic

class DriveTrainer(basic.Trainer):
    
    def get_train_loader(self, batch_size, trainset, sampler, **kwargs):
        train_loader =  create_carla_loader(
        trainset,
        input_size=data_config["input_size"],
        batch_size=batch_size,
        multi_view_input_size=args["multi_view_input_size"],
        is_training=True,
        scale=args["scale"],
        color_jitter=args["color_jitter"],
        interpolation=train_interpolation,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args["workers"],
        distributed=args["distributed"],
        collate_fn=collate_fn,
        pin_memory=args["pin_mem"],
    )
        return train_loader
    
    def get_test_loader(self, batch_size, testset, sampler, **kwargs):
        test_loader = create_carla_dataset(
            args.dataset,
            root=args.data_dir,
            towns=args.val_towns,
            weathers=args.val_weathers,
            batch_size=args.batch_size,
            with_lidar=args.with_lidar,
            with_seg=args.with_seg,
            with_depth=args.with_depth,
            multi_view=args.multi_view,
            augment_prob=args.augment_prob,
            temporal_frames=args.temporal_frames,
        )
        
        return test_loader
    
    def get_optimizer(self, model):
        linear_scaled_lr = (
        args.lr * args.batch_size * torch.distributed.get_world_size() / 512.0
        )
        args.lr = linear_scaled_lr
        if args.with_backbone_lr:
            if args.local_rank == 0:
                _logger.info(
                    "CNN backbone and transformer blocks using different learning rates!"
                )
            backbone_linear_scaled_lr = (
                args.backbone_lr
                * args.batch_size
                * torch.distributed.get_world_size()
                / 512.0
            )
            backbone_weights = []
            other_weights = []
            for name, weight in model.named_parameters():
                if "backbone" in name and "lidar" not in name:
                    backbone_weights.append(weight)
                else:
                    other_weights.append(weight)
            if args.local_rank == 0:
                _logger.info(
                    "%d weights in the cnn backbone, %d weights in other modules"
                    % (len(backbone_weights), len(other_weights))
                )
            optimizer = create_optimizer_v2(
                [
                    {"params": other_weights},
                    {"params": backbone_weights, "lr": backbone_linear_scaled_lr},
                ],
                **optimizer_kwargs(cfg=args),
            )
        else:
            optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
        
        return optimizer
    
    def train_model(self, config, trainset, sampler, **kwargs):
        batch_size = config["batch_size"]
        self.sampler = sampler
        tic = time.perf_counter()

        self.run_history.reset()

        self.train_run_start(config)
        self.callback_handler.call_event("on_train_run_start", self, config)

        self.train_loader = self.get_train_loader(batch_size, trainset, sampler)

        # Initializing the loss criterion
        self._loss_criterion = self.get_loss_criterion()

        # Initializing the optimizer
        self.optimizer = self.get_optimizer(self.model)
        self.lr_scheduler = self.get_lr_scheduler(config, self.optimizer)
        #self.optimizer = self._adjust_lr(config, self.lr_scheduler, self.optimizer)

        self.model.to(self.device)
        self.model.train()

        total_epochs = config["epochs"]

        for self.current_epoch in range(1, total_epochs + 1):
            self._loss_tracker.reset()
            self.train_epoch_start(config)
            self.callback_handler.call_event("on_train_epoch_start", self, config)

            train_metrics = self.train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fns,
                args,
                writer,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
            )

            #self.lr_scheduler_step()
            lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

            if hasattr(self.optimizer, "params_state_update"):
                self.optimizer.params_state_update()

            # Simulate client's speed
            if (
                self.client_id != 0
                and hasattr(Config().clients, "speed_simulation")
                and Config().clients.speed_simulation
            ):
                self.simulate_sleep_time()

            # Saving the model at the end of this epoch to a file so that
            # it can later be retrieved to respond to server requests
            # in asynchronous mode when the wall clock time is simulated
            if (
                hasattr(Config().server, "request_update")
                and Config().server.request_update
            ):
                self.model.cpu()
                training_time = time.perf_counter() - tic
                filename = f"{self.client_id}_{self.current_epoch}_{training_time}.pth"
                self.save_model(filename)
                self.model.to(self.device)

            self.run_history.update_metric("train_loss", self._loss_tracker.average)
            self.train_epoch_end(config)
            self.callback_handler.call_event("on_train_epoch_end", self, config)

        self.train_run_end(config)
        self.callback_handler.call_event("on_train_run_end", self, config)
    
    def test_model(self, config, testset, sampler=None, **kwargs):
        
        loader_eval = self.get_test_loader(config["batch_size"], testset, sampler)
        validate_loss_fns = {
        #"traffic": MVTL1Loss(1.0, l1_loss=l1_loss),
        "traffic": LAVLoss(),
        "waypoints": torch.nn.L1Loss(),
        "cls": cls_loss,
        "stop_cls": cls_loss,
            }
        eval_metrics = self.validate(
                epoch,
                model,
                loader_eval,
                validate_loss_fns,
                args,
                writer,
                amp_autocast=amp_autocast,
            )
        return eval_metrics
    
    def get_loss_criterion(self):
        train_loss_fns = {
        #"traffic": MVTL1Loss(1.0, l1_loss=l1_loss),
        "traffic": LAVLoss(),
        "waypoints": torch.nn.L1Loss(),
        "cls": cls_loss,
        "stop_cls": cls_loss,
    }
        return train_loss_fns
    
    def get_lr_scheduler(self, config, optimizer):
        return create_scheduler(args, optimizer)
    
    def train_one_epoch(
        self, 
        epoch,
        model,
        loader,
        optimizer,
        loss_fns,
        args,
        writer,
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_ema=None,
        mixup_fn=None,
    ):

        second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        losses_m = AverageMeter()
        losses_waypoints = AverageMeter()
        losses_traffic = AverageMeter()
        losses_velocity = AverageMeter()
        losses_traffic_light_state = AverageMeter()
        losses_stop_sign = AverageMeter()

        model.train()

        end = time.time()
        last_idx = len(loader) - 1
        num_updates = epoch * len(loader)
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            data_time_m.update(time.time() - end)
            if isinstance(input, (tuple, list)):
                batch_size = input[0].size(0)
            elif isinstance(input, dict):
                batch_size = input[list(input.keys())[0]].size(0)
            else:
                batch_size = input.size(0)
            if not args.prefetcher:
                if isinstance(input, (tuple, list)):
                    input = [x.cuda() for x in input]
                elif isinstance(input, dict):
                    for key in input:
                        if isinstance(input[key], list):
                            continue
                        input[key] = input[key].cuda()
                else:
                    input = input.cuda()
                if isinstance(target, (tuple, list)):
                    target = [x.cuda() for x in target]
                elif isinstance(target, dict):
                    for key in target:
                        target[key] = target[key].cuda()
                else:
                    target = target.cuda()

            with amp_autocast():
                output = model(input)
                loss_traffic, loss_velocity = loss_fns["traffic"](output[0], target[4])
                loss_waypoints = loss_fns["waypoints"](output[1], target[1])
                on_road_mask = target[2] < 0.5
                loss_traffic_light_state = loss_fns["cls"](
                    output[2], target[3]
                )
                loss_stop_sign = loss_fns["stop_cls"](output[3], target[6])
                loss = (
                    loss_traffic * 0.5
                    + loss_waypoints * 0.5
                    + loss_velocity * 0.05
                    + loss_traffic_light_state * 0.1
                    + loss_stop_sign * 0.01
                )

            if not args.distributed:
                losses_traffic.update(loss_traffic.item(), batch_size)
                losses_waypoints.update(loss_waypoints.item(), batch_size)
                losses_m.update(loss.item(), batch_size)

            optimizer.zero_grad()
            if loss_scaler is not None:
                loss_scaler(
                    loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(
                        model, exclude_head="agc" in args.clip_mode
                    ),
                    create_graph=second_order,
                )
            else:
                loss.backward(create_graph=second_order)
                if args.clip_grad is not None:
                    dispatch_clip_grad(
                        model_parameters(model, exclude_head="agc" in args.clip_mode),
                        value=args.clip_grad,
                        mode=args.clip_mode,
                    )
                optimizer.step()

            if model_ema is not None:
                model_ema.update(model)

            torch.cuda.synchronize()
            num_updates += 1
            batch_time_m.update(time.time() - end)
            if last_batch or batch_idx % args.log_interval == 0:
                lrl = [param_group["lr"] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    losses_m.update(reduced_loss.item(), batch_size)
                    reduced_loss_traffic = reduce_tensor(loss_traffic.data, args.world_size)
                    losses_traffic.update(reduced_loss_traffic.item(), batch_size)
                    reduced_loss_velocity = reduce_tensor(
                        loss_velocity.data, args.world_size
                    )
                    losses_velocity.update(reduced_loss_velocity.item(), batch_size)

                    reduced_loss_waypoints = reduce_tensor(
                        loss_waypoints.data, args.world_size
                    )
                    losses_waypoints.update(reduced_loss_waypoints.item(), batch_size)
                    reduced_loss_traffic_light_state = reduce_tensor(
                        loss_traffic_light_state.data, args.world_size
                    )
                    losses_traffic_light_state.update(
                        reduced_loss_traffic_light_state.item(), batch_size
                    )
                    reduced_loss_stop_sign = reduce_tensor(
                        loss_stop_sign.data, args.world_size
                    )
                    losses_stop_sign.update(reduced_loss_stop_sign.item(), batch_size)
                    if writer and args.local_rank == 0:
                        writer.add_scalar("train/loss", reduced_loss.item(), num_updates)
                        writer.add_scalar(
                            "train/loss_traffic", reduced_loss_traffic.item(), num_updates
                        )
                        writer.add_scalar(
                            "train/loss_velocity", reduced_loss_velocity.item(), num_updates
                        )
                        writer.add_scalar(
                            "train/loss_waypoints",
                            reduced_loss_waypoints.item(),
                            num_updates,
                        )
                        writer.add_scalar(
                            "train/loss_traffic_light_state",
                            reduced_loss_traffic_light_state.item(),
                            num_updates,
                        )
                        writer.add_scalar(
                            "train/loss_stop_sign",
                            reduced_loss_stop_sign.item(),
                            num_updates,
                        )

                        # Add Image
                        writer.add_image(
                            "train/front_view", retransform(input["rgb_front"][0]), num_updates
                        )
                        writer.add_image(
                            "train/left_view",
                            retransform(input["rgb_left"][0]),
                            num_updates,
                        )
                        writer.add_image(
                            "train/right_view",
                            retransform(input["rgb_right"][0]),
                            num_updates,
                        )
                        writer.add_image(
                            "train/rear_view",
                            retransform(input["rgb_rear"][0]),
                            num_updates,
                        )
                        writer.add_image(
                            "train/front_center_view",
                            retransform(input["rgb_center"][0]),
                            num_updates,
                        )
                        writer.add_image(
                            "train/pred_traffic",
                            torch.clip(output[0][0], 0, 1).view(1, 50, 50, 8)[:, :, :, 0],
                            num_updates,
                        )
                        writer.add_image(
                            "train/pred_traffic_render",
                            torch.clip(
                                torch.tensor(
                                    render(
                                        output[0][0].view(50, 50, 8).detach().cpu().numpy()
                                    )[:250, 25:275]
                                ),
                                0,
                                255,
                            ).view(1, 250, 250),
                            num_updates,
                        )
                        #input["lidar"][0] = input["lidar"][0] / torch.max(input["lidar"][0])
                        #writer.add_image(
                        #    "train/lidar", torch.clip(input["lidar"][0], 0, 1), num_updates
                        #)
                        writer.add_image(
                            "train/gt_traffic",
                            torch.clip(target[4][0], 0, 1).view(1, 50, 50, 8)[:, :, :, 0],
                            num_updates,
                        )
                        writer.add_image(
                            "train/gt_highres_traffic",
                            torch.clip(target[0][0], 0, 1),
                            num_updates,
                        )
                        writer.add_image(
                            "train/pred_waypoints",
                            torch.clip(
                                torch.tensor(
                                    render_waypoints(output[1][0].detach().cpu().numpy())[
                                        :250, 25:275
                                    ]
                                ),
                                0,
                                255,
                            ).view(1, 250, 250),
                            num_updates,
                        )
                        writer.add_image(
                            "train/gt_waypoints",
                            torch.clip(target[5][0], 0, 1),
                            num_updates,
                        )

                if args.local_rank == 0:
                    _logger.info(
                        "Train: {} [{:>4d}/{} ({:>3.0f}%)]  "
                        "Loss(traffic): {loss_traffic.val:>9.6f} ({loss_traffic.avg:>6.4f})  "
                        "Loss(waypoints): {loss_waypoints.val:>9.6f} ({loss_waypoints.avg:>6.4f})  "
                        "Loss(light): {loss_traffic_light_state.val:>9.6f} ({loss_traffic_light_state.avg:>6.4f})  "
                        "Loss: {loss.val:>9.6f} ({loss.avg:>6.4f})  "
                        "Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  "
                        "({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                        "LR: {lr:.3e}  "
                        "Data: {data_time.val:.3f} ({data_time.avg:.3f})".format(
                            epoch,
                            batch_idx,
                            len(loader),
                            100.0 * batch_idx / last_idx,
                            loss=losses_m,
                            loss_traffic=losses_traffic,
                            loss_waypoints=losses_waypoints,
                            loss_traffic_light_state=losses_traffic_light_state,
                            batch_time=batch_time_m,
                            rate=batch_size * args.world_size / batch_time_m.val,
                            rate_avg=batch_size * args.world_size / batch_time_m.avg,
                            lr=lr,
                            data_time=data_time_m,
                        )
                    )

                    if args.save_images and output_dir:
                        torchvision.utils.save_image(
                            input,
                            os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                            padding=0,
                            normalize=True,
                        )

            if (
                saver is not None
                and args.recovery_interval
                and (last_batch or (batch_idx + 1) % args.recovery_interval == 0)
            ):
                saver.save_recovery(epoch, batch_idx=batch_idx)

            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

            end = time.time()
            # end for

        if hasattr(optimizer, "sync_lookahead"):
            optimizer.sync_lookahead()

        return OrderedDict([("loss", losses_m.avg)])


    def validate(
        self, epoch, model, loader, loss_fns, args, writer, amp_autocast=suppress, log_suffix=""
        ):
        batch_time_m = AverageMeter()
        losses_m = AverageMeter()
        losses_waypoints = AverageMeter()
        losses_traffic = AverageMeter()
        losses_velocity = AverageMeter()
        losses_traffic_light_state = AverageMeter()
        losses_stop_sign = AverageMeter()

        l1_errorm = AverageMeter()
        traffic_light_state_errorm = AverageMeter()
        stop_sign_errorm = AverageMeter()

        model.eval()

        end = time.time()
        last_idx = len(loader) - 1
        with torch.no_grad():
            for batch_idx, (input, target) in enumerate(loader):
                last_batch = batch_idx == last_idx
                if isinstance(input, (tuple, list)):
                    batch_size = input[0].size(0)
                elif isinstance(input, dict):
                    batch_size = input[list(input.keys())[0]].size(0)
                else:
                    batch_size = input.size(0)
                if isinstance(input, (tuple, list)):
                    input = [x.cuda() for x in input]
                elif isinstance(input, dict):
                    for key in input:
                        input[key] = input[key].cuda()
                else:
                    input = input.cuda()
                if isinstance(target, (tuple, list)):
                    target = [x.cuda() for x in target]
                elif isinstance(target, dict):
                    for key in target:
                        input[key] = input[key].cuda()
                else:
                    target = target.cuda()

                with amp_autocast():
                    output = model(input)

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0 : target.size(0) : reduce_factor]

                loss_traffic, loss_velocity = loss_fns["traffic"](output[0], target[4])
                loss_waypoints = loss_fns["waypoints"](output[1], target[1])
                on_road_mask = target[2] < 0.5
                loss_traffic_light_state = loss_fns["cls"](output[2], target[3])
                loss_stop_sign = loss_fns["stop_cls"](output[3], target[6])
                loss = (
                    loss_traffic * 0.5
                    + loss_waypoints * 0.5
                    + loss_velocity * 0.05
                    + loss_traffic_light_state * 0.1
                    + loss_stop_sign * 0.01
                )

                traffic_light_state_error = accuracy(
                    output[2], target[3]
                )[0]
                stop_sign_error = accuracy(output[3], target[6])[0]

                if args.distributed:
                    reduced_loss = reduce_tensor(loss.data, args.world_size)
                    reduced_loss_traffic = reduce_tensor(loss_traffic.data, args.world_size)
                    reduced_loss_velocity = reduce_tensor(
                        loss_velocity.data, args.world_size
                    )
                    reduced_loss_waypoints = reduce_tensor(
                        loss_waypoints.data, args.world_size
                    )
                    reduced_loss_traffic_light_state = reduce_tensor(
                        loss_traffic_light_state.data, args.world_size
                    )
                    reduced_loss_stop_sign = reduce_tensor(
                        loss_stop_sign.data, args.world_size
                    )
                    reduced_traffic_light_state_error = reduce_tensor(
                        traffic_light_state_error, args.world_size
                    )
                    reduced_stop_sign_error = reduce_tensor(
                        stop_sign_error, args.world_size
                    )
                else:
                    reduced_loss = loss.data

                torch.cuda.synchronize()

                losses_m.update(reduced_loss.item(), batch_size)
                losses_traffic.update(reduced_loss_traffic.item(), batch_size)
                losses_velocity.update(reduced_loss_velocity.item(), batch_size)
                losses_waypoints.update(reduced_loss_waypoints.item(), batch_size)
                losses_traffic_light_state.update(
                    reduced_loss_traffic_light_state.item(), batch_size
                )
                losses_stop_sign.update(reduced_loss_stop_sign.item(), batch_size)

                l1_errorm.update(reduced_loss.item(), batch_size)
                traffic_light_state_errorm.update(
                    reduced_traffic_light_state_error.item(), batch_size
                )
                stop_sign_errorm.update(reduced_stop_sign_error.item(), batch_size)

                batch_time_m.update(time.time() - end)
                end = time.time()
                if args.local_rank == 0 and (
                    last_batch or batch_idx % args.log_interval == 0
                ):
                    log_name = "Test" + log_suffix
                    _logger.info(
                        "{0}: [{1:>4d}/{2}]  "
                        "Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  "
                        "Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  "
                        "Loss(traffic): {loss_traffic.val:>7.4f} ({loss_traffic.avg:>6.4f})  "
                        "Loss(waypoints): {loss_waypoints.val:>7.4f} ({loss_waypoints.avg:>6.4f})  "
                        "Loss(light): {loss_traffic_light_state.val:>9.6f} ({loss_traffic_light_state.avg:>6.4f})  "
                        "Acc(light): {traffic_light_state_errorm.val:>9.6f} ({traffic_light_state_errorm.avg:>6.4f})  ".format(
                            log_name,
                            batch_idx,
                            last_idx,
                            batch_time=batch_time_m,
                            loss_traffic_light_state=losses_traffic_light_state,
                            traffic_light_state_errorm=traffic_light_state_errorm,
                            loss=losses_m,
                            loss_traffic=losses_traffic,
                            loss_waypoints=losses_waypoints,
                        )
                    )
                    if writer:
                        # Add Image
                        writer.add_image(
                            "val/%d_front_view" % batch_idx,
                            retransform(input["rgb_front"][0]),
                            epoch,
                        )
                        writer.add_image(
                            "val/%d_left_view" % batch_idx,
                            retransform(input["rgb_left"][0]),
                            epoch,
                        )
                        writer.add_image(
                            "val/%d_right_view" % batch_idx,
                            retransform(input["rgb_right"][0]),
                            epoch,
                        )
                        writer.add_image(
                            "val/%d_front_center_view" % batch_idx,
                            retransform(input["rgb_center"][0]),
                            epoch,
                        )
                        writer.add_image(
                            "val/%d_rear_view" % batch_idx,
                            retransform(input["rgb_rear"][0]),
                            epoch,
                        )
                        writer.add_image(
                            "val/%d_pred_traffic" % batch_idx,
                            torch.clip(output[0][0], 0, 1).view(1, 50, 50, 8)[:, :, :, 0],
                            epoch,
                        )
                        writer.add_image(
                            "val/%d_gt_traffic" % batch_idx,
                            torch.clip(target[4][0], 0, 1).view(1, 50, 50, 8)[:, :, :, 0],
                            epoch,
                        )
                        writer.add_image(
                            "val/%d_highres_gt_traffic" % batch_idx,
                            torch.clip(target[0][0], 0, 1),
                            epoch,
                        )

                        writer.add_image(
                            "val/%d_gt_waypoints" % batch_idx,
                            torch.clip(target[5][0], 0, 1),
                            epoch,
                        )
                        writer.add_image(
                            "val/%d_pred_traffic_render" % batch_idx,
                            torch.clip(
                                torch.tensor(
                                    render(
                                        output[0][0].view(50, 50, 8).detach().cpu().numpy()
                                    )[:250, 25:275]
                                ),
                                0,
                                255,
                            ).view(1, 250, 250),
                            epoch,
                        )
                        writer.add_image(
                            "val/%d_pred_waypoints" % batch_idx,
                            torch.clip(
                                torch.tensor(
                                    render_waypoints(output[1][0].detach().cpu().numpy())[
                                        :250, 25:275
                                    ]
                                ),
                                0,
                                255,
                            ).view(1, 250, 250),
                            epoch,
                        )

            if writer:
                writer.add_scalar("val/loss", losses_m.avg, epoch)
                writer.add_scalar("val/loss_traffic", losses_traffic.avg, epoch)
                writer.add_scalar("val/loss_velocity", losses_velocity.avg, epoch)
                writer.add_scalar("val/loss_waypoints", losses_waypoints.avg, epoch)
                writer.add_scalar(
                    "val/loss_traffic_light_state", losses_traffic_light_state.avg, epoch
                )
                writer.add_scalar("val/loss_stop_sign", losses_stop_sign.avg, epoch)
                writer.add_scalar(
                    "val/acc_traffic_light_state", traffic_light_state_errorm.avg, epoch
                )
                writer.add_scalar("val/acc_stop_sign", stop_sign_errorm.avg, epoch)

        metrics = OrderedDict([("loss", losses_m.avg), ("l1_error", l1_errorm.avg)])

        return metrics



class WaypointL1Loss:
    def __init__(self, l1_loss=torch.nn.L1Loss):
        self.loss = l1_loss(reduction="none")
        self.weights = [
            0.1407441030399059,
            0.13352157985305926,
            0.12588535273178575,
            0.11775496498388233,
            0.10901991343009122,
            0.09952110967153563,
            0.08901438656870617,
            0.07708872007078788,
            0.06294267636589287,
            0.04450719328435308,
        ]

    def __call__(self, output, target):
        invaild_mask = target.ge(1000)
        output[invaild_mask] = 0
        target[invaild_mask] = 0
        loss = self.loss(output, target)  # shape: n, 12, 2
        loss = torch.mean(loss, (0, 2))  # shape: 12
        loss = loss * torch.tensor(self.weights, device=output.device)
        return torch.mean(loss)

class LAVLoss:
    def __init__(self):
        self.prob_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.loc_criterion = nn.L1Loss(reduction='none')
        self.ori_criterion = nn.L1Loss(reduction='none')
        self.box_criterion = nn.L1Loss(reduction='none')
        self.spd_criterion = nn.L1Loss(reduction='none')
        #self.loc_criterion = nn.SmoothL1Loss(reduction='none')
        #self.ori_criterion = nn.SmoothL1Loss(reduction='none')
        #self.box_criterion = nn.SmoothL1Loss(reduction='none')
        #self.spd_criterion = nn.SmoothL1Loss(reduction='none')

    def __call__(self, output, target):
        prob = target[:, : ,0:1]
        prob_mean = prob.mean()
        prob_mean = torch.maximum(prob_mean, torch.ones_like(prob_mean) * 1e-7)
        prob_det = torch.sigmoid(output[:, :, 0] * (1 - 2 * target[:, :, 0]))

        det_loss = (prob_det * self.prob_criterion(output[:, :, 0], target[:, :, 0])).mean() / prob_det.mean()
        loc_loss = (prob * self.loc_criterion(output[:, :, 1:3], target[:, :, 1:3])).mean() / prob_mean
        box_loss = (prob * self.box_criterion(output[:, :, 3:5], target[:, :, 3:5])).mean() / prob_mean
        ori_loss = (prob * self.ori_criterion(output[:, :, 5:7], target[:, :, 5:7])).mean() / prob_mean
        spd_loss = (prob * self.ori_criterion(output[:, :, 7:8], target[:, :, 7:8])).mean() / prob_mean

        det_loss = 0.4 * det_loss + 0.2 * loc_loss + 0.2 * box_loss + 0.2 * ori_loss
        return det_loss, spd_loss


class MVTL1Loss:
    def __init__(self, weight=1, l1_loss=torch.nn.L1Loss):
        self.loss = l1_loss()
        self.weight = weight

    def __call__(self, output, target):
        target_1_mask = target[:, :, 0].ge(0.01)
        target_0_mask = target[:, :, 0].le(0.01)
        target_prob_1 = torch.masked_select(target[:, :, 0], target_1_mask)
        output_prob_1 = torch.masked_select(output[:, :, 0], target_1_mask)
        target_prob_0 = torch.masked_select(target[:, :, 0], target_0_mask)
        output_prob_0 = torch.masked_select(output[:, :, 0], target_0_mask)
        if target_prob_1.numel() == 0:
            loss_prob_1 = 0
        else:
            loss_prob_1 = self.loss(output_prob_1, target_prob_1)
        if target_prob_0.numel() == 0:
            loss_prob_0 = 0
        else:
            loss_prob_0 = self.loss(output_prob_0, target_prob_0)
        loss_1 = 0.5 * loss_prob_0 + 0.5 * loss_prob_1

        output_1 = output[target_1_mask][:][:, 1:7]
        target_1 = target[target_1_mask][:][:, 1:7]
        if target_1.numel() == 0:
            loss_2 = 0
        else:
            loss_2 = self.loss(target_1, output_1)

        # speed pred loss
        output_2 = output[target_1_mask][:][:, 7]
        target_2 = target[target_1_mask][:][:, 7]
        if target_2.numel() == 0:
            loss_3 = target_2.sum() # torch.tensor([0.0]).cuda()
        else:
            loss_3 = self.loss(target_2, output_2)
        return 0.5 * loss_1 * self.weight + 0.5 * loss_2, loss_3