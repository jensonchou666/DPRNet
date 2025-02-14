from jscv.models.pred_stat_c import result_kargs_hard_trunc


'''
    从训练好的模型进行PTDR:
    python train.py {model}.py -c {dataset}.py config/dataset_repair/ex_PTDR.py -r ?/?

'''


repair_dataset_args = dict(
    do_save = True,
    save_dir = './data/gid_water_train_repaired',
    classes_trunc = [0.9, 0.97],
    save_label = True,
    save_org_img = False,
)

pred_stat_sample_size       =   10
pred_stat_start_epoch       =   1  #? 至少40轮后再用此文件
pred_stat_count_list_len    =   100
pred_stat_min_value         =   0.6



def setting_before(cfg):
    D1 = result_kargs_hard_trunc
    D1['wrong_only'] = False
    batchs = len(cfg.train_dataset) // cfg.train_batch_size
    print("Prediction Truncation based Dataset Repair (PTDR) Enable")

    sample_size = cfg.pred_stat_sample_size
    queue_blocks = max(batchs // sample_size, 1)
    
    cfg.repair_dataset_args['batchs'] = batchs

    cfg.pred_statistic_args = dict(
        start_epoch=cfg.pred_stat_start_epoch,
        fig_max_lines=10,
        count_len=cfg.pred_stat_count_list_len,
        min_value=cfg.pred_stat_min_value,
        sample_capacity=batchs,
        queue_blocks=queue_blocks,
        result_kargs=D1,
        save_per_x_blocks=1,
        repair_dataset_args=cfg.repair_dataset_args
        )
    
    if cfg.repair_dataset_args['do_save']:
        print("Save repair-dataset to", cfg.repair_dataset_args['save_dir'])


def before_create_model(cfg):
    setting_before(cfg)
    if 'save_images_args' in cfg:
        cfg['save_images_args']['per_n_step'] = -1   # 禁用存图
    cfg.dprnet_args.update(
        dict(
            do_prediction_statistics=True, #耦合在网络里了，没改
            cfg_file=cfg,
        )
    )


def after_create_model(cfg):
    model = cfg.model
    model.pred_logits = True
    import jscv.utils.analyser as ana
    ana.set_rgb([ana.黑, ana.蓝])



def get_evaluators(cfg):
    from jscv.utils.trainer import SegmentEvaluator, Joint_Evaluator
    return Joint_Evaluator(
        SegmentEvaluator(cfg.num_classes, cfg.classes),
    )
