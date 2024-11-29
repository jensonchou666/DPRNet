from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary, unary_from_softmax,unary_from_labels
import numpy as np
import pydensecrf.densecrf as dcrf

def dense_crf_2d(img, output_probs):
    # img 为H*W*C 的原图，output_probs 为 输出概率 sigmoid 输出（h，w），#seg_map - 假设为语义分割的 mask, hxw, np.array 形式.

    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    img = np.ascontiguousarray(img)

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=20, compat=3)
    d.addPairwiseBilateral(sxy=30, srgb=20, rgbim=img, compat=10)

    Q = d.inference(5)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q

def dense_crf(img, processed_probabilities):
    
    softmax = processed_probabilities
    # processed_probabilities：CNN 预测概率 经过 softmax [n_label,H,W]
    C, H, W = softmax.shape

    unary = unary_from_softmax(softmax)
    #2.自己生成一元势函数
    # The inputs should be C-continious -- we are using Cython wrapper
    # unary = -np.log(output_probs)
    # unary = unary.reshape((C, -1))
    # unary = np.ascontiguousarray(unary)  # (C, n)

    d = dcrf.DenseCRF(H * W, C)  # h,w,n_class

    d.setUnaryEnergy(unary)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    
    # Pairwise potentials（二元势）
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                      img=img, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    # 快捷方法
    # d.addPairwiseGaussian(sxy=3, compat=3)
    # d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=img, compat=10)
    # 迭代次数,对于IMG_1702(2592*1456)这张图,迭代5 16.807087183s 迭代20 37.5700438023s
    Q = d.inference(5)
    res = np.argmax(Q, axis=0).reshape((img.shape[0], img.shape[1]))

    return res



from jscv.hr_models.pathes_segmentor import *




class DenseCrfNet(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    
    def forward(self, batch:dict):
        img = batch['img']  # B,C,H,W
        assert isinstance(img, torch.Tensor)

        result = self.model(batch)
        
        pred = result['pred']   # softmax    B,C,H,W
        
        pred_2 = []
        for imgi, predi in zip(img, pred):
            imgi = imgi.permute(1,2,0).cpu().numpy()
            predi = predi.cpu().numpy()
            predi = dense_crf(imgi, predi)
            pred_2.append(torch.from_numpy(predi))
        result['pred'] = torch.stack(pred_2)
        return result
    
    def predict(self, img, zero_img=False):
        pred = self.model.predict(img, zero_img)
        pred_argmax = pred.argmax(1)
        pred_2 = []
        for imgi, predi in zip(img, pred):
            imgi = imgi.permute(1,2,0).cpu().numpy()
            predi = predi.cpu().numpy()
            predi = dense_crf(imgi, predi)
            pred_2.append(torch.from_numpy(predi))
        
        return torch.stack(pred_2), pred_argmax



def test_mIOU_on_val_dataset(save_dir='a/densecrf_val'):
    import os
    import tqdm
    from jscv.utils.utils import warmup
    from jscv.utils.load_checkpoint import load_checkpoint
    from jscv.utils.trainer import SegmentEvaluator
    ckpt = 'work_dir/GID_Water/seg_r101_d4-e80/final/epoch=61@val_mIoU=86.87@seg_r101_d4-GID_Water.ckpt'

    os.environ['CUDA_VISIBLE_DEVICES']='0'
    torch.set_grad_enabled(False)
    
    net = ResSegNet('resnet101', decoder_args=fpn_decoder_args_1)
    model = PatchesSegmentor(net, val_setting=((1,1), 1/4))
    load_checkpoint(model, ckpt)
    model = DenseCrfNet(model)
    model = model.cuda().eval()
    warmup()
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ds = water_seg_dataset('data/gid_water/val')
    eval = SegmentEvaluator(2, ['other', 'water'], from_logits=False)
    eval_coarse = SegmentEvaluator(2, ['other', 'water'], from_logits=False)

    print("begin:")
    t0 = time()
    tq = tqdm.tqdm(total=ds.len)

    for fi, (img, label, id) in enumerate(ds.batch()):
        pred, coarse_pred = model.predict(img.cuda())

        # cv2.imwrite(save_dir+f"/{fi}_0_img_{id}.jpg", 
        #             cv2.cvtColor(img[0].permute(1,2,0).cpu().numpy(), cv2.COLOR_BGR2RGB))
        cv2.imwrite(save_dir+f"/{fi}_1_mask_{id}.png", 
                    label[0].cpu().numpy().astype(np.int8)*255)
        cv2.imwrite(save_dir+f"/{fi}_2_coarse_{id}.png", 
                    coarse_pred[0].cpu().numpy().astype(np.int8)*255)
        cv2.imwrite(save_dir+f"/{fi}_3_refine_{id}.png", 
                    pred[0].cpu().numpy().astype(np.int8)*255)

        eval.append({'pred': pred.cpu(), 'target_cpu': label})
        eval_coarse.append({'pred': coarse_pred.cpu(), 'target_cpu': label})

        tq.update(1)
    eval.evaluate()
    eval_coarse.evaluate()
    print("coarse:", eval_coarse)
    print("refine:", eval)
        

def test_predict_speed_pure():
    '''
        计算纯粹的推理时间，
    '''
    import os
    from jscv.utils.utils import warmup
    from jscv.utils.load_checkpoint import load_checkpoint
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    epochs = 1
    input = torch.zeros(1, 3, 1024*7, 1024*7)
    load_gpu = True
    
    net = ResSegNet('resnet101', decoder_args=fpn_decoder_args_1)
    model = PatchesSegmentor(net, val_setting=((1,1), 1/4))
    load_checkpoint(model, 'work_dir/GID_Water/seg_r101_d4-e80/final/epoch=61@val_mIoU=86.87@seg_r101_d4-GID_Water.ckpt')
    model = DenseCrfNet(model)


    torch.set_grad_enabled(False)
    if load_gpu:
        input = input.cuda()
    model = model.cuda().eval()
    warmup()


    count_1.DO_DEBUG = False
    import tqdm
    
    print("begin:")
    t0 = time()

    for i in tqdm.tqdm(range(epochs)):
        model.predict(input, zero_img=True)

    t0 = (time()-t0)/epochs
    print("FPS:", 1/t0, f'   {(t0*100):.2f}ms')


if __name__ == "__main__":
    
    test_mIOU_on_val_dataset()
