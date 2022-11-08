from utils.utils import square_distance
import torch


def test_square_distance():
    Xb = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    Xq = torch.tensor([[4.0, 5.0, 6.0], [10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])

    assert torch.equal(
        square_distance(Xq, Xb),
        torch.tensor(
            [[27.0, 0.0, 27.0], [243.0, 108.0, 27.0], [432.0, 243.0, 108.0], [675.0, 432.0, 243.0]]
        ),
    )
