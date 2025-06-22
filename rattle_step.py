import torch
from torch.autograd.functional import jacobian
class Manifold:
    def __init__(self, constraints):
        self.constraints = constraints

    def G(self, x):
        return torch.stack([g(x) for g in self.constraints], 1)


    def J(self, x, create_graph):
        return (2*x).unsqueeze(1)



    def rattle_step(self,x, v1, h, create_graph=False):
        '''
        Defining a function to take a step in the position, velocity form.
        g should be a vector-valued function of constraints.
        :return: x_1, v_1
        '''
        DV = torch.zeros_like(x)
        batch_size, dim = x.shape
        M = torch.eye(x.shape[-1], device='cuda').repeat(x.shape[0],1,1)
        DV_col = DV.reshape(batch_size,-1, 1)


        x_col = x.reshape(batch_size,-1, 1)
        v1_col = v1.reshape(batch_size,-1, 1)

        # doing Newton-Raphson iterations
        x2 = x_col + h * v1_col - 0.5*(h**2)* torch.bmm(M, DV_col)
        Q_col = x2
        Q = Q_col.squeeze(-1)


        J1 = self.J(x_col.reshape(batch_size, dim),create_graph)

        diff = torch.tensor([1.]).cuda()
        steps =0

        for _ in range(10):
            J2 = self.J(Q.reshape(batch_size, dim),create_graph)
            R = torch.bmm(torch.bmm(J2,M),J1.mT)
            dL = torch.bmm(torch.linalg.pinv(R),self.G(Q).unsqueeze(-1))
            diff = torch.bmm(torch.bmm(M,J1.mT), dL)
            Q= Q- diff.squeeze(-1)
            steps +=1

        # half step for velocity
        Q_col = Q.reshape(batch_size,-1,1)
        v1_half = (Q_col - x_col)/h
        x_col = Q_col
        J1 = self.J(x_col.reshape(batch_size, dim), create_graph)

        J2 = self.J(Q.reshape(batch_size, dim),create_graph)
        P = torch.bmm(torch.bmm(J1, M),J1.mT)
        T = torch.bmm(J1, (2/h * v1_half - torch.bmm(M, DV_col)))

        #solving the linear system
        L = torch.linalg.solve(P,T)

        v1_col = v1_half - h/2 * DV_col - h/2 * torch.bmm(J2.mT,L)

        return x_col.reshape(batch_size, dim), v1_col.reshape(batch_size, dim)
