from jscv.utils.trainer import *

import scipy

def matte_mad(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    error_mad = np.mean(np.abs(pred_matte - gt_matte))    
    return error_mad



def matte_sad(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    error_sad = np.sum(np.abs(pred_matte - gt_matte))    
    return error_sad

def matte_mse(pred_matte, gt_matte):
    assert (len(pred_matte.shape) == len(gt_matte.shape))
    error_mse = np.mean(np.power(pred_matte - gt_matte, 2))    
    return error_mse

def matte_grad(pred_matte, gt_matte):
    assert(len(pred_matte.shape) == len(gt_matte.shape))
    # alpha matte 的归一化梯度，标准差 =1.4，1 阶高斯导数的卷积
    predict_grad = scipy.ndimage.filters.gaussian_filter(pred_matte, 1.4, order=1) 
    gt_grad = scipy.ndimage.filters.gaussian_filter(gt_matte, 1.4, order=1)
    error_grad = np.sum(np.power(predict_grad - gt_grad, 2))
    return error_grad


class MatteEvaluator(Evaluator):

    def __init__(self):
        super().__init__()
        self.mad = []
        self.sad = []
        self.mse = []
        self.grad = []


    def append(self, result):
        pred = result["pred"].detach().cpu().numpy()
        target_cpu = result["target_cpu"].numpy()
        self.mad.append(matte_mad(pred, target_cpu))
        self.sad.append(matte_sad(pred, target_cpu))
        self.mse.append(matte_mse(pred, target_cpu))
        self.grad.append(matte_grad(pred, target_cpu))
    
    
    def evaluate(self):
        D = {
            f'{self.stat}_MAD': sum(self.mad) / len(self.mad),
            f'{self.stat}_SAD': sum(self.sad) / len(self.sad),
            f'{self.stat}_MSE': sum(self.mse) / len(self.mse),
            f'{self.stat}_Grad': sum(self.grad) / len(self.grad),
        }

        self.mad = []
        self.sad = []
        self.mse = []
        self.grad = []

        self.result = D
        return D

    def __str__(self) -> str:
        if self.result is None:
            return 'None result'
        else:
            return str(self.result)
