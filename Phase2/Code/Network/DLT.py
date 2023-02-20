import torch
import torch.nn as nn
import cv2
import numpy as np

class TensorDlt(torch.nn.Module):
    def __init__(self, source_points, target_points):
        super().__init__()
        self.source_points = source_points
        self.target_points = target_points
        self.H = None

    def forward(self, x, x_p):
        B_size = x.shape[0]
        x = x.view(-1, 4, 2)
        x_p = x_p.view(-1, 4, 2)
        u_i, v_i = x.transpose(1, 2)
        u_i_p, v_i_p = x_p.transpose(1, 2)

        A = torch.zeros(B_size, 8, 8, dtype=torch.float64)
        b = torch.zeros(B_size, 8, 1, dtype=torch.float64)

        idx_x = torch.tensor([0, 2, 4, 6], dtype=torch.float64)
        idx_y = torch.tensor([1, 3, 5, 7], dtype=torch.float64)
        zeros = torch.zeros(B_size, 4, dtype=torch.float64)
        ones = torch.ones(B_size, 4, dtype=torch.float64)

        A[:, idx_x, :] = torch.stack([zeros, zeros, zeros, -u_i, -v_i, -ones, v_i_p*u_i, v_i_p*v_i], dim=2).long()
        A[:, idx_y, :] = torch.stack([u_i, v_i, ones, zeros, zeros, zeros, -u_i_p*u_i, -u_i_p*v_i], dim=2).long()
        b[:, idx_x, :] = -v_i_p.unsqueeze(-1)
        b[:, idx_y, :] = u_i_p.unsqueeze(-1)

        ret = torch.linalg.pinv(A) @ b
        ones_2 = torch.ones(B_size, 1, 1, dtype=torch.float64)
        ret = torch.cat([ret, ones_2], dim=1)
        ret = ret.reshape(B_size, 3, 3)
        self.H = ret
        return ret


def test_tensorDLT():
    actual_H4pt_list = [
                            [
                                [0,   0, 128 , 0 , 0 , 128 , 128 ,  128],
                                [-213, -181, -219, 219, -15, -212, 15, 312],
                            ],
                            [
                                [-23, -18, 99, 29, -5, 106, 133, 160],
                                [-233, -118, -229, 291, -513, -21, 52, -20],
                            ]

                    ]

    H4pt_src = np.array(actual_H4pt_list[0][0]).reshape(4,2)
    print(f'src:{H4pt_src}')
    H4pt_dst = np.array(actual_H4pt_list[1][0]).reshape(4,2)
    print(f'dst:{H4pt_dst}')

    H_cv2,_ = cv2.findHomography(H4pt_src,H4pt_dst)

    H4pt_torch = torch.tensor(actual_H4pt_list,dtype=torch.float64)

    # test_img = torch.zeros(size=(2,128,128))
    H_torch = TensorDlt(H4pt_torch[0],H4pt_torch[1])
    H_torch.forward(H4pt_torch[0],H4pt_torch[1])
    print(f'cv2 homography: {H_cv2}')
    print(f'torch homography: {H_torch.H}')


test_tensorDLT()