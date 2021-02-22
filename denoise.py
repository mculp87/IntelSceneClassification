import numpy as np

from sporco import util
from sporco import linalg
from sporco import plot
plot.config_notebook_plotting()
from sporco.cupy import (cupy_enabled, np2cp, cp2np, select_device_by_load,
                         gpu_info)
from sporco.cupy.admm import cbpdn
from sporco.cupy import cnvrep as cr
from sporco.cupy import linalg as cplinalg
import sporco.cupy.linalg as sl
import sporco.cupy.linalg as sp
from sporco.cupy.linalg import irfftn,rfftn
from sporco.cupy.linalg import inner

import torch
import torch.fft as tfft
from torch.nn.functional import pad as tpad
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# npd = 16
# fltlmbd = 200.0
# lmbda = 5.0e-2
# ropt = cbpdn.ConvBPDN.Options({'Verbose': False,
#                               'DataType': np.float32,
#                               'MaxMainIter': 50,
#                               'HighMemSolve': True,
#                               'RelStopTol': 1e-7,
#                               'NonNegCoef': True,
#                               'FastSolve': True,
#                               'rho': 1e3,
#                               'AutoRho': {"Enabled": False}})

def mytake(T, idx, axis):
    slices = (slice(None),) * axis + (idx, Ellipsis)
    return T[slices]

def solvemdbi_ism(ah, rho, b, axisM, axisK):
    r"""Solve a multiple diagonal block linear system with a scaled
    identity term by iterated application of the Sherman-Morrison
    equation.

    The computation is performed in a way that avoids explictly
    constructing the inverse operator, leading to an :math:`O(K^2)`
    time cost.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

    .. math::
      (\rho I + \mathbf{a}_0 \mathbf{a}_0^H + \mathbf{a}_1
       \mathbf{a}_1^H + \; \ldots \; + \mathbf{a}_{K-1}
       \mathbf{a}_{K-1}^H) \; \mathbf{x} = \mathbf{b}

    where each :math:`\mathbf{a}_k` is an :math:`M`-vector.
    The sums, inner products, and matrix products in this equation are
    taken along the :math:`M` and :math:`K` axes of the corresponding
    multi-dimensional arrays; the solutions are independent over the
    other axes.

    Parameters
    ----------
    ah : array_like
      Linear system component :math:`\mathbf{a}^H`
    rho : float
      Linear system parameter :math:`\rho`
    b : array_like
      Linear system component :math:`\mathbf{b}`
    axisM : int
      Axis in input corresponding to index m in linear system
    axisK : int
      Axis in input corresponding to index k in linear system

    Returns
    -------
    x : ndarray
      Linear system solution :math:`\mathbf{x}`
    """

    if axisM < 0:
        axisM += len(ah.shape)
    if axisK < 0:
        axisK += len(ah.shape)

    K = ah.shape[axisK]
    a = torch.conj(ah)
    gamma = torch.zeros(a.shape, device=device, dtype=a.dtype)
    dltshp = list(a.shape)
    dltshp[axisM] = 1
    delta = torch.zeros(dltshp, device=device, dtype=a.dtype)
    slcnc = (slice(None),) * axisK
    aidx = (slice(None),)
    alpha = mytake(a, [0], axisK) / rho
    beta = b / rho

    del b
    for k in range(0, K):

        slck = slcnc + (slice(k, k + 1), Ellipsis)
        gamma[slck] = alpha
        delta[slck] = 1.0 + torch.sum(ah[slck] * gamma[slck], dim=axisM, keepdim=True)

        d = gamma[slck] * torch.sum(ah[slck] * beta, dim=axisM, keepdim=True)
        beta -= d / delta[slck]

        if k < K - 1:
            alpha[:] = mytake(a, [k + 1], axisK) / rho
            for l in range(0, k + 1):
                slcl = slcnc + (slice(l, l + 1),)
                d = gamma[slcl] * torch.sum(ah[slcl] * alpha, dim=axisM, keepdim=True)
                alpha -= d / delta[slcl]

    return beta

def tikhonov_filter(s, *, lmbda=1.0, npd=16, dtype=torch.float32):
    r"""Lowpass filter based on Tikhonov regularization.

    Lowpass filter image(s) and return low and high frequency
    components, consisting of the lowpass filtered image and its
    difference with the input image. The lowpass filter is equivalent to
    Tikhonov regularization with `lmbda` as the regularization parameter
    and a discrete gradient as the operator in the regularization term,
    i.e. the lowpass component is the solution to

    .. math::
      \mathrm{argmin}_\mathbf{x} \; (1/2) \left\|\mathbf{x} - \mathbf{s}
      \right\|_2^2 + (\lambda / 2) \sum_i \| G_i \mathbf{x} \|_2^2 \;\;,

    where :math:`\mathbf{s}` is the input image, :math:`\lambda` is the
    regularization parameter, and :math:`G_i` is an operator that
    computes the discrete gradient along image axis :math:`i`. Once the
    lowpass component :math:`\mathbf{x}` has been computed, the highpass
    component is just :math:`\mathbf{s} - \mathbf{x}`.

    Parameters
    ----------
    s : array_like
      Input image or array of images.
    lmbda : float
      Regularization parameter controlling lowpass filtering.
    npd : int, optional (default=16)
      Number of samples to pad at image boundaries.

    Returns
    -------
    slp : array_like
      Lowpass image or array of images.
    shp : array_like
      Highpass image or array of images.
    """

    grv = torch.from_numpy(np.array([-1.0, 1.0]).reshape([2, 1])).to(s.device)
    gcv = torch.from_numpy(np.array([-1.0, 1.0]).reshape([1, 2])).to(s.device)
    fftopt = {"s": (s.shape[0] + 2*npd, s.shape[1] + 2*npd), "dim": (0,1)}
    Gr = tfft.rfftn(grv, **fftopt)
    Gc = tfft.rfftn(gcv, **fftopt)
    A = 1.0 + lmbda * (torch.conj(Gr)*Gr + torch.conj(Gc)*Gc).real
    if s.ndim > 2:
      A = A[(slice(None),)*2 + (np.newaxis,)*(s.ndim-2)]
    fill = ((npd, npd),)*2 + ((0, 0),)*(s.ndim-2)
    snp = np.pad(s.cpu().numpy(), fill, 'symmetric')
    # sp = tpad(s, ((npd, npd),)*2 + ((0, 0),)*(s.ndim-2), 'symmetric')
    sp = torch.from_numpy(snp).to(s.device)
    # sp = torch.from_numpy(np.pad(s.numpy(), ((npd, npd),)*2 + ((0, 0),)*(s.ndim-2), 'symmetric'))
    spshp = sp.shape
    sp = tfft.rfftn(sp, dim=(0, 1))
    sp /= A
    sp = tfft.irfftn(sp, s=spshp[0:2], dim=(0, 1))
    slp = sp[npd:(sp.shape[0] - npd), npd:(sp.shape[1] - npd)]
    shp = s - slp
    return slp, shp

def rrs(ax, b):
    r"""Relative residual of the solution to a linear equation.

    The standard relative residual for the linear system
    :math:`A \mathbf{x} = \mathbf{b}` is :math:`\|\mathbf{b} - A
    \mathbf{x}\|_2 / \|\mathbf{b}\|_2`. This function computes a
    variant :math:`\|\mathbf{b} - A \mathbf{x}\|_2 /
    \max(\|A\mathbf{x}\|_2, \|\mathbf{b}\|_2)` that is robust to
    the case :math:`\mathbf{b} = 0`.

    Parameters
    ----------
    ax : array_like
      Linear component :math:`A \mathbf{x}` of equation
    b : array_like
      Constant component :math:`\mathbf{b}` of equation

    Returns
    -------
    x : float
      Relative residual
    """

    nrm = max(torch.linalg.norm(ax.ravel(), ord=2),
              torch.linalg.norm(b.ravel(), ord=2))
    if nrm == 0.0:
        return 0.0
    else:
        return torch.linalg.norm((ax - b).ravel(), ord=2) / nrm

def prox_l1(v, alpha):
    r"""Compute the proximal operator of the :math:`\ell_1` norm (scalar
    shrinkage/soft thresholding)

     .. math::
      \mathrm{prox}_{\alpha f}(\mathbf{v}) =
      \mathcal{S}_{1,\alpha}(\mathbf{v}) = \mathrm{sign}(\mathbf{v})
      \odot \max(0, |\mathbf{v}| - \alpha)

    where :math:`f(\mathbf{x}) = \|\mathbf{x}\|_1`.

    Unlike the corresponding :func:`norm_l1`, there is no need for an
    `axis` parameter since the proximal operator of the :math:`\ell_1`
    norm is the same when taken independently over each element, or
    over their sum.


    Parameters
    ----------
    v : array_like
      Input array :math:`\mathbf{v}`
    alpha : float or array_like
      Parameter :math:`\alpha`

    Returns
    -------
    x : ndarray
      Output array
    """
    return torch.sign(v) * (torch.clip(torch.abs(v) - alpha, 0, float('Inf')))

def solvedbi_sm_c(ah, a, rho, axis=4):
    r"""Compute cached component used by :func:`solvedbi_sm`.

    Parameters
    ----------
    ah : array_like
      Linear system component :math:`\mathbf{a}^H`
    a : array_like
      Linear system component :math:`\mathbf{a}`
    rho : float
      Linear system parameter :math:`\rho`
    axis : int, optional (default 4)
      Axis along which to solve the linear system

    Returns
    -------
    c : torch.Tensor
      Argument :math:`\mathbf{c}` used by :func:`solvedbi_sm`
    """

    return ah / (torch.sum(ah * a, dim=axis) + rho)

def solvedbi_sm(ah, rho, b, c=None, axis=4):
    r"""Solve a diagonal block linear system with a scaled identity term
    using the Sherman-Morrison equation.

    The solution is obtained by independently solving a set of linear
    systems of the form (see :cite:`wohlberg-2016-efficient`)

    .. math::
      (\rho I + \mathbf{a} \mathbf{a}^H ) \; \mathbf{x} = \mathbf{b} \;\;.

    In this equation inner products and matrix products are taken along
    the specified axis of the corresponding multi-dimensional arrays; the
    solutions are independent over the other axes.

    Parameters
    ----------
    ah : array_like
      Linear system component :math:`\mathbf{a}^H`
    rho : float
      Linear system parameter :math:`\rho`
    b : array_like
      Linear system component :math:`\mathbf{b}`
    c : array_like, optional (default None)
      Solution component :math:`\mathbf{c}` that may be pre-computed using
      :func:`solvedbi_sm_c` and cached for re-use.
    axis : int, optional (default 4)
      Axis along which to solve the linear system

    Returns
    -------
    x : ndarray
      Linear system solution :math:`\mathbf{x}`
    """

    a = torch.conj(ah)
    if c is None:
        c = solvedbi_sm_c(ah, a, rho, axis)
    return (b - (a * inner(c, b, axis=axis))) / rho

# Need to remake SPORCO ConvBPDN to be more compatible with pytorch
class CSC(object):
    devopt = {
        "device": device,
        "non_blocking": True
    }
    _instance = None

    def __init__(self,
                dictfname,
                batchdims, *,
                dimK=None,
                dimN=2,
                dtype=torch.float32,
                lmbda=None,
                **kwargs):
      self.dtype = dtype

      self.tensoropt = {
          "device": device,
          "dtype": self.dtype
      }

      # torch.cuda.ipc_collect()

      # Convert this into a static class variable
      batchkeys = {"sigshape", "channels", "batchsize"}

      # Loading dictionary and torch.tensor version
      with open(dictfname, 'rb') as dctfile:
        self.dictionary = np.load(dctfile)
      D = torch.tensor(self.dictionary, **self.tensoropt)
      Dshape = tuple(D.shape)

      assert all(key in batchkeys for key in batchdims.keys()), \
        "Batch dimensions needs to inclue keywords: " + str(batchkeys)
      Sshape = tuple(batchdims["sigshape"]) + (batchdims["channels"], batchdims["batchsize"])


      assert Dshape[dimN] == Sshape[dimN], "Channels between dictionary and" + \
        " images must be consistent."

      # Infer problem dimensions and set relevant attributes of self
      if not hasattr(self, 'cri'):
        self.cri = CSC_ConvRepIndexing(Dshape, Sshape, dimK=dimK, dimN=dimN)
      print(self.cri)
      self.cri.npToTns = (self.cri.axisK, self.cri.axisC,) + tuple(self.cri.axisN)
      self.cri.tnsToNp = tuple(range(2,2+self.cri.dimN)) + (1, 0, 2+self.cri.dimN)

      self.fftopt = {
        "s": self.cri.Nv,
        "dim": self.cri.axisN
      }

      # Preallocating memory
      self.S = torch.zeros(self.cri.shpS, **self.tensoropt)
      self.Sf = torch.fft.rfftn(self.S, **self.fftopt)
      self.D = D.reshape(self.cri.shpD)
      self.Df = torch.fft.rfftn(self.D, **self.fftopt)
      self.DSf = torch.conj(self.Df) * self.Sf
      self.X = torch.zeros(self.cri.shpX, **self.tensoropt)
      self.Xf = torch.fft.rfftn(self.X, **self.fftopt)

      self.Nx = np.prod(self.cri.shpX)
      self.Y = torch.zeros(self.cri.shpX, **self.tensoropt)
      self.Ypre = torch.zeros(self.cri.shpX, **self.tensoropt)
      self.U = torch.zeros(self.cri.shpX, **self.tensoropt)
      self.YU = torch.zeros(self.Y.shape, **self.tensoropt)

      lmbda = 5.0e-2
      self.opt = cbpdn.ConvBPDN.Options({
        'Verbose': False,
        'DataType': np.float32,
        'MaxMainIter': 50,
        'HighMemSolve': True,
        'RelStopTol': 1e-7,
        'NonNegCoef': True,
        'FastSolve': True,
        'rho': 1e3,
        'AutoRho': {"Enabled": False}
      })

      self.tikopt = {
        "lmbda": 200.0,
        "npd": 16
      }

      self.lmbda = lmbda if lmbda else 1.0

      self.rho = self.opt.get('rho', 50.0*self.lmbda + 1.0)

      # Initialise attributes representing penalty parameter and other
      # parameters
      autorho = self.opt.get('AutoRho', {})
      self.rho_tau = autorho.get('Scaling', 2.0)
      self.rho_mu = autorho.get('RsdlRatio', 10.0)
      self.rlx = self.opt.get('RelaxParam', 1.0)

      # Set rho_xi attribute (see Sec. VI.C of wohlberg-2015-adaptive)
      self.rho_xi = autorho.get('RsdlTarget')
      if self.rho_xi == None:
          if self.lmbda != 0.0:
              self.rho_xi = float((1.0 + (18.3)**(np.log10(self.lmbda) + 1.0)))
          else:
              self.rho_xi = 1.0

      self.Nc = self.Nx

      self.iter = 0

    def reset(self):
      self.X.fill_(0.0)
      self.Xf.fill_(0.0)

      self.Y.fill_(0.0)
      self.Ypre.fill_(0.0)
      self.U.fill_(0.0)
      self.YU.fill_(0.0)

      self.rho = self.opt.get('rho', 50.0*self.lmbda + 1.0)

      # Initialise attributes representing penalty parameter and other
      # parameters
      autorho = self.opt.get('AutoRho', {})
      self.rho_tau = autorho.get('Scaling', 2.0)
      self.rho_mu = autorho.get('RsdlRatio', 10.0)
      self.rlx = self.opt.get('RelaxParam', 1.0)

      # Set rho_xi attribute (see Sec. VI.C of wohlberg-2015-adaptive)
      self.rho_xi = autorho.get('RsdlTarget')
      if self.rho_xi == None:
          if self.lmbda != 0.0:
              self.rho_xi = float((1.0 + (18.3)**(np.log10(self.lmbda) + 1.0)))
          else:
              self.rho_xi = 1.0

      self.Nc = self.Nx

      self.iter = 0

    @staticmethod
    def batchdims(sigshape, channels, batchsize):
      batchdims = {
        "sigshape": tuple(sigshape),
        "channels": channels,
        "batchsize": batchsize
      }
      return batchdims

    def setimage(self, S):
      self.reset()
      if S.shape[-1] < self.cri.K:
        self.S[...,:S.shape[-1],:] = S.to(**self.devopt).reshape(S.shape + (1,))
        self.S[...,S.shape[-1]:,:].fill_(0.0)
      else:
        self.S = S.to(**self.devopt).reshape(self.cri.shpS)
      # self.S = S.to(**self.devopt).reshape(self.cri.shpS)
      self.Sf = torch.fft.rfftn(self.S, **self.fftopt)

      self.setdict()

    def setdict(self, D=None):
      """Set dictionary array."""
      # Change the dictionary and its Fourier transform
      if D:
        self.D = D.device(device, non_blocking=True)
        self.Df = torch.fft.rfftn(self.D, **self.tensoropt)

      # Compute D^H S
      self.DSf = torch.conj(self.Df) * self.Sf
      if self.cri.Cd > 1:
        self.DSf = torch.sum(self.DSf, dim=self.cri.axisC, keepdim=True)
      if self.opt['HighMemSolve'] and self.cri.Cd == 1:
        self.c = solvedbi_sm_c(self.Df,
                                torch.conj(self.Df),
                                self.rho,
                                self.cri.axisM)
      else:
        self.c = None

    def ystep(self):
      r"""Minimise Augmented Lagrangian with respect to
      :math:`\mathbf{y}`."""

      self.Y = prox_l1(self.AX + self.U, (self.lmbda / self.rho))

      if self.opt['NonNegCoef']:
        self.Y[self.Y < 0.0] = 0.0
      if self.opt['NoBndryCross']:
        for n in range(0, self.cri.dimN):
            self.Y[(slice(None),) * n + (slice(1 - self.D.shape[n], None),)] = 0.0

    def xstep(self):
      r"""Minimise Augmented Lagrangian with respect to
      :math:`\mathbf{x}`."""

      self.YU[:] = self.Y - self.U

      b = self.DSf + self.rho * torch.fft.rfftn(self.YU, **self.fftopt)
      if self.cri.Cd == 1:
        self.Xf[:] = solvedbi_sm(self.Df,
                                    self.rho,
                                    b,
                                    self.c,
                                    self.cri.axisM)
      else:
        self.Xf[:] = solvemdbi_ism(self.Df,
                                    self.rho,
                                    b,
                                    self.cri.axisM,
                                    self.cri.axisC)

      self.X = torch.fft.irfftn(self.Xf, **self.fftopt)

      if self.opt['LinSolveCheck']:
        Dop = lambda x: torch.sum(self.Df * x, dim=self.cri.axisM)
        if self.cri.Cd == 1:
            DHop = lambda x: torch.conj(self.Df) * x
        else:
            DHop = lambda x: torch.sum(torch.conj(self.Df) * x,
                                      dim=self.cri.axisC)
        ax = DHop(Dop(self.Xf)) + self.rho * self.Xf
        self.xrrs = rrs(ax, b)
      else:
        self.xrrs = None

    def ustep(self):
        """Dual variable update."""

        self.U += self.AX - self.Y
    
    def rhochange(self):
        """Updated cached c array when rho changes."""

        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = solvedbi_sm_c(self.Df, torch.conj(self.Df), self.rho,
                                      self.cri.axisM)
    
    def reconstruct(self, X=None, out=None, devopt=None):
        if X is None: X = self.Y
        if devopt is None: devopt = self.devopt
        Xf = torch.fft.rfftn(X, **self.fftopt)
        Sf = torch.sum(self.Df * Xf, axis=self.cri.axisM)
        return torch.fft.irfftn(Sf, **self.fftopt).to(**devopt)
    
    def getcoef(self):
        return self.Y
    
    def relax_AX(self):
        self.AXnr = self.X
        if self.rlx == 1.0:
            self.AX = self.X
        else:
            self.AX = self.rlx*self.X + (1.0 - self.rlx)*self.Y

    def solve(self, S=None, out=None, *, devopt=None, normed=False, lmbda=None):
        if S is not None:
          if normed:
            T = 0.5*S.permute((2,3,1,0)).to(**self.devopt) + 0.5
            Sl, Sh = tikhonov_filter(T, **self.tikopt)
          else:
            Sh = S
          Sh.to(**self.devopt)
          # Assuming Tensor image input, could modify to check if numpy or tensor.
          self.setimage(Sh)
          self.iter = 0
        
        if devopt is None:
          devopt = self.devopt
        else:
          self.devopt = devopt

        if lmbda:
          self.lmbda = lmbda
        
        for k in range(self.opt.get('MaxMainIter', 10)):
          self.Ypre = torch.clone(self.Y)

          self.xstep()

          self.relax_AX()

          self.ystep()

          self.ustep()

          if self.opt['AutoRho', 'Enabled'] or not self.opt['FastSolve']:
            r, s, epri, edua = self.compute_residuals()
            self.update_rho(self.k, r, s)
            if r < epri and s < edua:
              break
        self.iter += k+1

        # Sh = self.reconstruct(devopt=devopt)
        # T = (2.0*(Sl + Sh) - 1.0).squeeze().permute((3,2,0,1))
        if normed:
          Shrecon = self.reconstruct().squeeze()
          Srecon = Sl + Shrecon
          T = 2.0*Srecon.permute((3,2,0,1)) - 1.0
        else:
          T = self.reconstruct()
        return T

    def normsolve(self, S):
      return self.solve(S, normed=True)

    def rsdl_r(self, AX, Y):
        """Compute primal residual vector.

        Overriding this method is required if methods :meth:`cnst_A`,
        :meth:`cnst_AT`, :meth:`cnst_B`, and :meth:`cnst_c` are not
        overridden.
        """

        return AX - Y

    def rsdl_s(self, Yprev, Y):
        """Compute dual residual vector."""

        return self.rho * (Yprev - Y)
    
    def rsdl_rn(self, AX, Y):
        """Compute primal residual normalisation term."""

        return max((torch.linalg.norm(AX.ravel(), ord=2),
                    torch.linalg.norm(Y.ravel(), ord=2)))

    def rsdl_sn(self, U):
        """Compute dual residual normalisation term."""

        return self.rho * torch.linalg.norm(U.ravel(), ord=2)

    def compute_residuals(self):
        """Compute residuals and stopping thresholds."""

        if self.opt['AutoRho', 'StdResiduals']:
            r = torch.linalg.norm(self.rsdl_r(self.AXnr, self.Y))
            s = torch.linalg.norm(self.rsdl_s(self.Yprev, self.Y))
            epri = np.sqrt(self.Nc) * self.opt['AbsStopTol'] + \
                self.rsdl_rn(self.AXnr, self.Y) * self.opt['RelStopTol']
            edua = np.sqrt(self.Nx) * self.opt['AbsStopTol'] + \
                self.rsdl_sn(self.U) * self.opt['RelStopTol']
        else:
            rn = self.rsdl_rn(self.AXnr, self.Y)
            if rn == 0.0:
                rn = 1.0
            sn = self.rsdl_sn(self.U)
            if sn == 0.0:
                sn = 1.0
            r = torch.linalg.norm(self.rsdl_r(self.AXnr, self.Y).ravel(), ord=2) / rn
            s = torch.linalg.norm(self.rsdl_s(self.Yprev, self.Y).ravel(), ord=2) / sn
            epri = np.sqrt(self.Nc) * self.opt['AbsStopTol'] / rn + \
                self.opt['RelStopTol']
            edua = np.sqrt(self.Nx) * self.opt['AbsStopTol'] / sn + \
                self.opt['RelStopTol']

        return r, s, epri, edua
      

import pprint
class CSC_ConvRepIndexing(object):
    """Array dimensions and indexing for CSC problems.

    Manage the inference of problem dimensions and the roles of
    :class:`numpy.ndarray` indices for convolutional representations in
    convolutional sparse coding problems (e.g.
    :class:`.admm.cbpdn.ConvBPDN` and related classes).
    """

    def __init__(self, Dshape, Sshape, dimK=None, dimN=2):
        """Initialise a ConvRepIndexing object.

        Initialise a ConvRepIndexing object representing dimensions
        of S (input signal), D (dictionary), and X (coefficient array)
        in a convolutional representation.  These dimensions are
        inferred from the input `D` and `S` as well as from parameters
        `dimN` and `dimK`.  Management and inferrence of these problem
        dimensions is not entirely straightforward because
        :class:`.admm.cbpdn.ConvBPDN` and related classes make use
        *internally* of S, D, and X arrays with a standard layout
        (described below), but *input* `S` and `D` are allowed to
        deviate from this layout for the convenience of the user.

        The most fundamental parameter is `dimN`, which specifies the
        dimensionality of the spatial/temporal samples being
        represented (e.g. `dimN` = 2 for representations of 2D
        images).  This should be common to *input* S and D, and is
        also common to *internal* S, D, and X.  The remaining
        dimensions of input `S` can correspond to multiple channels
        (e.g. for RGB images) and/or multiple signals (e.g. the array
        contains multiple independent images).  If input `S` contains
        two additional dimensions (in addition to the `dimN` spatial
        dimensions), then those are considered to correspond, in
        order, to channel and signal indices.  If there is only a
        single additional dimension, then determination whether it
        represents a channel or signal index is more complicated.  The
        rule for making this determination is as follows:

        * if `dimK` is set to 0 or 1 instead of the default ``None``,
          then that value is taken as the number of signal indices in
          input `S` and any remaining indices are taken as channel
          indices (i.e. if `dimK` = 0 then dimC = 1 and if `dimK` = 1
          then dimC = 0).
        * if `dimK` is ``None`` then the number of channel dimensions is
          determined from the number of dimensions in the input
          dictionary `D`. Input `D` should have at least `dimN` + 1
          dimensions, with the final dimension indexing dictionary
          filters. If it has exactly `dimN` + 1 dimensions then it is a
          single-channel dictionary, and input `S` is also assumed to be
          single-channel, with the additional index in `S` assigned as a
          signal index (i.e. dimK = 1). Conversely, if input `D` has
          `dimN` + 2 dimensions it is a multi-channel dictionary, and
          the additional index in `S` is assigned as a channel index
          (i.e. dimC = 1).

        Note that it is an error to specify `dimK` = 1 if input `S`
        has `dimN` + 1 dimensions and input `D` has `dimN` + 2
        dimensions since a multi-channel dictionary requires a
        multi-channel signal. (The converse is not true: a
        multi-channel signal can be decomposed using a single-channel
        dictionary.)

        The *internal* data layout for S (signal), D (dictionary), and
        X (coefficient array) is (multi-channel dictionary)
        ::

            sptl.          chn  sig  flt
          S(N0,  N1, ...,  C,   K,   1)
          D(N0,  N1, ...,  C,   1,   M)
          X(N0,  N1, ...,  1,   K,   M)

        or (single-channel dictionary)

        ::

            sptl.          chn  sig  flt
          S(N0,  N1, ...,  C,   K,   1)
          D(N0,  N1, ...,  1,   1,   M)
          X(N0,  N1, ...,  C,   K,   M)

        where

        * Nv = [N0, N1, ...] and N = N0 x N1 x ... are the vector of sizes
          of the spatial/temporal indices and the total number of
          spatial/temporal samples respectively
        * C is the number of channels in S
        * K is the number of signals in S
        * M is the number of filters in D

        It should be emphasised that dimC and `dimK` may take on values
        0 or 1, and represent the number of channel and signal
        dimensions respectively *in input S*. In the internal layout
        of S there is always a dimension allocated for channels and
        signals. The number of channel dimensions in input `D` and the
        corresponding size of that index are represented by dimCd
        and Cd respectively.

        Parameters
        ----------
        D : array_like
          Input dictionary
        S : array_like
          Input signal
        dimK : 0, 1, or None, optional (default None)
          Number of dimensions in input signal corresponding to multiple
          independent signals
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions of signal samples
        """

        # Determine whether dictionary is single- or multi-channel
        self.dimCd = len(Dshape) - (dimN + 1)
        if self.dimCd == 0:
            self.Cd = 1
        else:
            self.Cd = Dshape[-2]

        # Numbers of spatial, channel, and signal dimensions in
        # external S are dimN, dimC, and dimK respectively. These need
        # to be calculated since inputs D and S do not already have
        # the standard data layout above, i.e. singleton dimensions
        # will not be present
        if dimK is None:
            rdim = len(Sshape) - dimN
            if rdim == 0:
                (dimC, dimK) = (0, 0)
            elif rdim == 1:
                dimC = self.dimCd  # Assume S has same number of channels as D
                dimK = len(Sshape) - dimN - dimC  # Assign remaining channels to K
            else:
                (dimC, dimK) = (1, 1)
        else:
            dimC = len(Sshape) - dimN - dimK  # Assign remaining channels to C

        self.dimN = dimN  # Number of spatial dimensions
        self.dimC = dimC  # Number of channel dimensions in S
        self.dimK = dimK  # Number of signal dimensions in S

        # Number of channels in S
        if self.dimC == 1:
            self.C = Sshape[dimN]
        else:
            self.C = 1
        Cx = self.C - self.Cd + 1

        # Ensure that multi-channel dictionaries used with a signal with a
        # matching number of channels
        if self.Cd > 1 and self.C != self.Cd:
            raise ValueError("Multi-channel dictionary with signal with "
                             "mismatched number of channels (Cd=%d, C=%d)" %
                             (self.Cd, self.C))

        # Number of signals in S
        if self.dimK == 1:
            self.K = Sshape[self.dimN + self.dimC]
        else:
            self.K = 1

        # Number of filters
        self.M = Dshape[-1]

        # Shape of spatial indices and number of spatial samples
        self.Nv = Sshape[0:dimN]
        self.N = np.prod(np.array(self.Nv))

        # Axis indices for each component of X and internal S and D
        self.axisN = tuple(range(0, dimN))
        self.axisC = dimN
        self.axisK = dimN + 1
        self.axisM = dimN + 2

        # Shapes of internal S, D, and X
        self.shpD = Dshape[0:dimN] + (self.Cd,) + (1,) + (self.M,)
        self.shpS = self.Nv + (self.C,) + (self.K,) + (1,)
        self.shpX = self.Nv + (Cx,) + (self.K,) + (self.M,)

    def __str__(self):
        """Return string representation of object."""

        return pprint.pformat(vars(self))