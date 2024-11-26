import oneflow as flow

BATCH_SIZE = 16
BROADCAST = [flow.sbp.broadcast]
P0 = flow.placement("cuda", ranks=[0])
P1 = flow.placement("cuda", ranks=[1])

class Stage0Module(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = flow.nn.Flatten()
        self.linear0 = flow.nn.Linear(28*28, 512)
        self.relu0 = flow.nn.ReLU()

    def forward(self, x):
        out = self.flatten(x)
        out = self.linear0(out)
        out = self.relu0(out)
        return out

class Stage1Module(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = flow.nn.Linear(512, 512)
        self.relu1 = flow.nn.ReLU()
        self.linear2 = flow.nn.Linear(512, 10)
        self.relu2 = flow.nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        return out

class PipelineModule(flow.nn.Module):
    def __init__(self):
        super().__init__()
        self.m_stage0 = Stage0Module()
        self.m_stage1 = Stage1Module()

        self.m_stage0.to_global(placement=P0, sbp=BROADCAST)
        self.m_stage1.to_global(placement=P1, sbp=BROADCAST)

    def forward(self, x):
        out_stage0 = self.m_stage0(x)
        in_stage1 = out_stage0.to_global(placement=P1, sbp=BROADCAST)
        out_stage1 = self.m_stage1(in_stage1)
        return out_stage1

module_pipeline = PipelineModule()
sgd = flow.optim.SGD(module_pipeline.parameters(), lr=0.001)

class PipelineGraph(flow.nn.Graph):
    def __init__(self):
        super().__init__()
        self.module_pipeline = module_pipeline
        self.module_pipeline.m_stage0.to(nn.graph.GraphModule).set_stage(stage_id=0, placement=P0)
        self.module_pipeline.m_stage1.to(nn.graph.GraphModule).set_stage(stage_id=1, placement=P1)
        self.loss_fn = flow.nn.CrossEntropyLoss()
        self.config.set_gradient_accumulation_steps(2)
        self.add_optimizer(sgd)

    def build(self, x, y):
        out = self.module_pipeline(x)
        loss = self.loss_fn(out, y)
        loss.backward()
        return loss

graph_pipeline = PipelineGraph()

x = flow.randn(BATCH_SIZE, 1, 28, 28)
x = x.to_global(P0, BROADCAST)
y = flow.randint(0, 10, (BATCH_SIZE,))
y = y.to_global(P1, BROADCAST)

for i in range(20):
    loss = graph_pipeline(x, y)
    print(loss.to_local())