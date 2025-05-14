import numpy as np
from models.model_base import ModelBase
from models.base.init_utils import weights_init
from configs import *
from utils.network_utils import get_network
from utils.data_utils import get_dataloader
from pruner.pruning import *
from train_test import *
import time

def main():
    config = init_config()
    logger, writer = init_logger(config)

    state = None
    # ===== build/load model =====
    if config.pretrained:
        state = torch.load(config.pretrained)
        model = state['net']
        masks = state['mask']
        config.send_mail_str += f"use pre-trained mode -> acc:{state['acc']} epoch:{state['epoch']}\n"
        config.network = state['args'].network
        config.depth = state['args'].depth
        config.dataset = state['args'].dataset
        config.batch_size = state['args'].batch_size
        config.learning_rate = state['args'].learning_rate
        config.weight_decay = state['args'].weight_decay
        config.epoch = state['args'].epoch
        config.target_ratio = state['args'].target_ratio
        print('load model finish')
        print(state['args'])
    else:
        model = get_network(config.network, config.depth, config.dataset, use_bn=config.get('use_bn', True))
        masks = None


    mb = ModelBase(config.network, config.depth, config.dataset, model)
    mb.cuda()



    # ===== get dataloader =====
    trainloader, testloader = get_dataloader(config.dataset, config.batch_size, 256, 4, root=config.dp)


    # ===== pruning =====
    if masks is None:
        logger.info('** Target ratio: %.5f' % (config.target_ratio))
        mb.model.apply(weights_init)
        print("=> Applying weight initialization(%s)." % config.get('init_method', 'kaiming'))
        masks, _ = Pruner(mb, trainloader, 'cuda',  config)
    mb.register_mask(masks)


    print_inf = print_mask_information(mb, logger)
    config.send_mail_str += print_inf
    pr_str, rem_ratio, eff_ratio = masks_compare(mb, masks, trainloader, 'cuda')
    config.send_mail_str += pr_str
    logger.info('  LR: %.5f, WD: %.5f, Epochs: %d' % (config.learning_rate, config.weight_decay, config.epoch))
    config.send_mail_str += 'LR: %.5f, WD: %.5f, Epochs: %d, Batch: %d \n' % (config.learning_rate, config.weight_decay, config.epoch, config.batch_size)
    if abs((1-config.target_ratio)*100 - rem_ratio) >= 1:
        _str = "ERROR: Pruning ratio not as expected\n"
        print(_str)
        config.send_mail_str += _str
        config.epoch = 1
    if eff_ratio == 0 or eff_ratio < rem_ratio*0.05:  # The effective compression rate is too low, no training
        _str = "ERROR: Effective compression ratio is too low\n"
        print(_str)
        config.send_mail_str += _str
        config.epoch = 1
    # ===== storage mask =====
    if config.storage_mask == 1:
        state = {
            'args': config,
            'mask': masks,
        }
        path = os.path.join(config.checkpoint_dir, config.exp_name)
        torch.save(state, path)
        print("=> storage mask finish", config.exp_name)
        exit()

    # ===== train =====
    start_time = time.time()
    tr_str, print_inf = train_once(mb, trainloader, testloader, config, writer, logger, state, config.lr_mode, config.optim_mode, masks=masks)#
    end_time = time.time();
    elapsed_time = end_time - start_time
    print(f"操作耗时：{elapsed_time}秒")
    config.send_mail_str += print_inf
    config.send_mail_str += tr_str
    print(config.send_mail_str)


if __name__ == '__main__':
    main()
