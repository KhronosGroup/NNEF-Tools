import torch
import torch.nn.functional as F
from torch.onnx import register_custom_op_symbolic


def aim_affine_grid(g, trans, shape, align_corners):
    return g.op("com.example::affine_grid", trans, shape, align_corners)


def aim_grid_sample(g, input, grid, mode, padding, align_corners):
    return g.op("com.example::grid_sample", input, grid, mode, padding, align_corners)


register_custom_op_symbolic('::affine_grid_generator', aim_affine_grid, 1)
register_custom_op_symbolic('::grid_sampler', aim_grid_sample, 1)


class AffineTransform(torch.nn.Module):

    def __init__(self, width, height):
        super(AffineTransform, self).__init__()
        self.width = width
        self.height = height

    def forward(self, input, theta):
        batch = int(input.shape[0])     # int() forces static shape instead of dynamic Shape() op
        channel = int(input.shape[1])
        grid = F.affine_grid(size=[batch, channel, self.height, self.width], theta=theta)
        return F.grid_sample(input, grid)


class Model(torch.nn.Module):

    def __init__(self, grid_size):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(4, 4), stride=(2, 2))
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(3, 3), stride=(1, 1))
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.affine = AffineTransform(width=grid_size[0], height=grid_size[1])

    def forward(self, x, t):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.affine(x, t)
        x = self.conv2(x)
        return x


model = Model(grid_size=(100, 100))
model.eval()

x = torch.randn(1, 3, 224, 224, requires_grad=False)
y = torch.zeros(size=(1, 2, 3))

torch.onnx.export(model, (x, y), "test.onnx", opset_version=10)
