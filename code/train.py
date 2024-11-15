import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import imageio
from option import args
from models import *
from loss import *
import utility
import math
from feature_transformation import pooled_spatial_similarity, spatial_similarity, channel_similarity, batch_similarity, FSP, AT, MMD, gaussian_rbf
from datetime import datetime
import MyDataloader

torch.manual_seed(100)
torch.cuda.manual_seed(100)
os.environ['CUDA_VISIBLE_DEVICES']=str(args.gpu_id)
device = torch.device('cpu' if args.cpu else 'cuda')
torch.backends.cudnn.enabled = False


best = 0
    
def load_teachers():
    print("Loading Teacher ====================================>")
    teachers = []
    
    if "EDSR" in args.teacher:
        args.n_feats = 256
        args.n_resblocks = 32
        args.res_scale = 0.1
        net = edsr.EDSR(args).to(device)
        if int(args.scale[0]) == 2:
            print("loading EDSRx2")
            net.load_state_dict_teacher(torch.load('../teacher_checkpoint/EDSR_BIX2.pt'))
        elif int(args.scale[0]) == 3:
            print("loading EDSRx3")
            net.load_state_dict_teacher(torch.load('../teacher_checkpoint/EDSR_BIX3.pt'))
        elif int(args.scale[0]) == 4:
            print("loading EDSRx4")
            net.load_state_dict_teacher(torch.load('../teacher_checkpoint/EDSR_BIX4.pt'))
        elif int(args.scale[0]) == 8:
            print("loading EDSRx8")
            net.load_state_dict_student(torch.load('../teacher_checkpoint/EDSR_BIX8.pt'))
        if args.precision == 'half':
            net.half()
        teachers.append(net)
    
    if "RCAN" in args.teacher:
        args.n_resblocks = 20
        args.n_resgroups = 10
        net = rcan.RCAN(args).to(device)
        if args.scale == 2:
            print("loading RCANx2")
            net.load_state_dict_teacher(torch.load('../teacher_checkpoint/RCAN_BIX2.pt'))
        elif args.scale == 3:
            print("loading RCANx3")
            net.load_state_dict_teacher(torch.load('../teacher_checkpoint/RCAN_BIX3.pt'))
        elif args.scale == 4:
            print("loading RCANx4")
            net.load_state_dict_teacher(torch.load('../teacher_checkpoint/RCAN_BIX4.pt'))
        elif args.scale == 8:
            print("loading RCANx8")
            net.load_state_dict_teacher(torch.load('../teacher_checkpoint/RCAN_BIX8.pt'))
        if args.precision == 'half':
            net.half()
        teachers.append(net)
    
    if "SAN" in args.teacher:
        args.n_resblocks = 10
        args.n_resgroups = 20
        args.n_feats = 64
        net = san.SAN(args).to(device)
        if args.scale == 2:
            print("loading SANx2")
            net.load_state_dict_teacher(torch.load('../teacher_checkpoint/SAN_BIX2.pt'))
        elif args.scale == 3:
            print("loading SANx3")
            net.load_state_dict_teacher(torch.load('../teacher_checkpoint/SAN_BIX3.pt'))
        elif args.scale == 4:
            print("loading SANx4")
            net.load_state_dict_teacher(torch.load('../teacher_checkpoint/SAN_BIX4.pt'))
        elif args.scale == 8:
            print("loading SANx8")
            net.load_state_dict_teacher(torch.load('../teacher_checkpoint/SAN_BIX8.pt'))
        if args.precision == 'half':
            net.half()
        teachers.append(net)
    
    
    for teacher in teachers:
        for p in teacher.parameters():
            p.requires_grad = False
    
    return teachers
    
    
def create_student_model():
    print("Preparing Student ===================================>")
    student_checkpoint = utility.checkpoint(args)
    if args.model == 'EDSR':
        args.n_resblocks = 16
        args.n_feats = 64
        args.res_scale = 1.0
        student = edsr.EDSR(args).to(device)
    elif args.model == 'RCAN':
        args.n_resblocks = 6
        args.n_resgroups = 10
        student = rcan.RCAN(args).to(device)
    elif args.model == 'SAN':
        args.n_resblocks = 10
        args.n_resgroups = 6
        args.n_feats = 64
        student = san.SAN(args).to(device)
    elif args.model == 'RDN':
        student = rdn.RDN(args).to(device)
    
    if args.precision == 'half':
        student.half()
    if args.resume:
        load_from = os.path.join(student_checkpoint.dir, 'model', 'model_latest.pt')
        student.load_state_dict_student(torch.load(load_from))
    return student_checkpoint, student
    
def prepare_criterion():
    criterion = Loss(args, student_ckp)
    if args.resume:
        criterion.load(student_ckp.dir)
    return criterion

def prepare_optimizer():
    optimizer = utility.make_optimizer(args, student)
    if args.resume:
        optimizer.load(student_ckp.dir, epoch=len(student_ckp.log))
    return optimizer



def teacher_selector(teachers):
    idx = np.random.randint(len(teachers))
    return teachers[idx]

def prepare(lr, hr):
    def _prepare(tensor):
        if args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(lr), _prepare(hr)]


def train(epoch):
    optimizer.schedule()
    criterion.step()
    student.train()
    criterion.start_log()
    
    lr = optimizer.get_lr()

    student_ckp.write_log(
        '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, lr)
    )
    
    timer_data, timer_model = utility.timer(), utility.timer()
    for batch, (lr, hr, _, __) in enumerate(train_loader):
        
        lr, hr = prepare(lr, hr)
        timer_data.hold()
        timer_model.tic()
        
        optimizer.zero_grad()
        student_fms, student_sr = student(lr)
        teacher = teacher_selector(teachers)
        teacher_fms, teacher_sr = teacher(lr)

 
        student_fms = [student_fms[ind] for ind in eval(args.features)]
        teacher_fms = [teacher_fms[ind] for ind in eval(args.features)]
        
        aggregated_student_fms = []
        aggregated_teacher_fms = []
        
        if args.feature_loss_used == 1:	
            for distillation_type in args.feature_distilation_type.split('+'):
                if 'PSA' in distillation_type:
                    aggregated_student_fms.append([pooled_spatial_similarity(fm, args.pool_size, args.pool_type) for fm in student_fms])
                    aggregated_teacher_fms.append([pooled_spatial_similarity(fm, args.pool_size, args.pool_type) for fm in teacher_fms])
                elif 'MMD' in distillation_type:
                    aggregated_student_fms.append([MMD(fm, args.pool_size, args.pool_type) for fm in student_fms])
                    aggregated_teacher_fms.append([MMD(fm, args.pool_size, args.pool_type) for fm in teacher_fms])
                elif 'RBF' in distillation_type:
                    aggregated_student_fms.append([gaussian_rbf(fm, args.pool_size, args.P, args.rbf_gamma, args.pool_type) for fm in student_fms])
                    aggregated_teacher_fms.append([gaussian_rbf(fm, args.pool_size, args.P, args.rbf_gamma, args.pool_type) for fm in teacher_fms])
                elif 'SA' in distillation_type:
                    aggregated_student_fms.append([spatial_similarity(fm) for fm in student_fms])
                    aggregated_teacher_fms.append([spatial_similarity(fm) for fm in teacher_fms])
                elif 'CA' in distillation_type:
                    aggregated_student_fms.append([channel_similarity(fm) for fm in student_fms])
                    aggregated_teacher_fms.append([channel_similarity(fm) for fm in teacher_fms])
                elif 'IA' in distillation_type:
                    aggregated_student_fms.append([batch_similarity(fm) for fm in student_fms])
                    aggregated_teacher_fms.append([batch_similarity(fm) for fm in teacher_fms])
                elif 'FSP' in distillation_type:
                    aggregated_student_fms.append([FSP(student_fms[i], student_fms[i+1]) for i in range(len(student_fms) - 1)])
                    aggregated_teacher_fms.append([FSP(teacher_fms[i], teacher_fms[i+1]) for i in range(len(teacher_fms) - 1)])
                elif 'AT' in distillation_type:
                    aggregated_student_fms.append([AT(fm) for fm in student_fms])
                    aggregated_teacher_fms.append([AT(fm) for fm in teacher_fms])
                elif 'fitnet' in distillation_type:
                    aggregated_student_fms.append([fm for fm in student_fms])
                    aggregated_teacher_fms.append([fm for fm in teacher_fms])
        
        # print(student_fms[0].shape, aggregated_student_fms[0][0].shape)
        total_loss = criterion(student_sr, teacher_sr, hr, aggregated_student_fms, aggregated_teacher_fms)
            
        total_loss.backward()
        optimizer.step()
        stu_psn = utility.calc_psnr(utility.quantize(student_sr, args.rgb_range), hr, args.scale, args.rgb_range)
        tea_psn = utility.calc_psnr(utility.quantize(teacher_sr, args.rgb_range), hr, args.scale, args.rgb_range)
        # stu_psn = calculate_psnr(utility.quantize(student_sr, args.rgb_range), hr, args.scale)
        # tea_psn = calculate_psnr(utility.quantize(teacher_sr, args.rgb_range), hr, args.scale)
        timer_model.hold()

        with torch.no_grad():
            if batch == 1000 and (epoch == 1 or epoch % 5 == 0):
                stu_img = student_sr[0].permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
                tea_img = teacher_sr[0].permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
                hr_img = hr[0].permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
                lr_img = lr[0].permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
                imageio.imwrite("./train_samples/train_" + str(epoch) + "_stu_sample.jpg", stu_img)
                imageio.imwrite("./train_samples/train_" + str(epoch) + "_tea_sample.jpg", tea_img)
                imageio.imwrite("./train_samples/train_" + str(epoch) + "_hr_sample.jpg", hr_img)
                imageio.imwrite("./train_samples/train_" + str(epoch) + "_lr_sample.jpg", lr_img)
            if (batch) % args.print_every == 0:
                student_ckp.write_log('{} [{}/{}]\t{}\t{:.1f}+{:.1f}s, student psn:{:.2f}, teacher psn:{:.2f}'.format(
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            (batch) * args.batch_size,
                            len(train_loader.dataset),
                            criterion.display_loss(batch),
                            timer_model.release(),
                            timer_data.release(),
                            stu_psn,
                            tea_psn))

        timer_data.tic()    
    
    criterion.end_log(len(train_loader.dataset))


def valid(epoch):
    student.eval()
    tot_psn = 0
    with torch.no_grad():
        # if args.save_results:
        #     student_ckp.begin_background()
        student_ckp.write_log('\nEvaluation:')
        student_ckp.add_log(torch.zeros(1, len(valid_loader)))

        timer_test = utility.timer()
        for batch, (lr, hr, lr_path, __) in enumerate(valid_loader):
            lr, hr = prepare(lr, hr)
            fms, sr = student(lr)
            sr = utility.quantize(sr, args.rgb_range)

            save_list = [sr]
            psn = utility.calc_psnr(sr, hr, args.scale, args.rgb_range)
            # psn = calculate_psnr(sr, hr, args.scale)
            tot_psn += psn
            if args.save_gt:
                save_list.extend([lr, hr])

            # if args.save_results:
            #     student_ckp.save_results(args.data_train, lr_path[0], save_list, args.scale)

            # best = student_ckp.log.max(0)

            if batch == 1000 and (epoch == 1 or epoch % 5 == 0):
                stu_img = sr[0].permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
                hr_img = hr[0].permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
                lr_img = lr[0].permute((1, 2, 0)).cpu().numpy().astype(np.uint8)
                imageio.imwrite("./train_samples/valid_" + str(epoch) + "_stu_sample.jpg", stu_img)
                imageio.imwrite("./train_samples/valid_" + str(epoch) + "_hr_sample.jpg", hr_img)
                imageio.imwrite("./train_samples/valid_" + str(epoch) + "_lr_sample.jpg", lr_img)
            if (batch) % args.print_every == 0:
                student_ckp.write_log(
                    '[{}/{}][x{}]\tPSNR: {:.3f} ave: {:.3f}'.format(
                        (batch) * args.batch_size,
                        len(valid_loader.dataset),
                        args.scale,
                        psn,
                        tot_psn / (batch+1),
                    )
                )
        student_ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        student_ckp.write_log('Saving...')

        # if args.save_results:
        #     student_ckp.end_background()

        global best
        if best < tot_psn / (len(valid_loader)):
            save(is_best=True, epoch=epoch)
            best = tot_psn / (len(valid_loader))
            print("new best: ", best)
        else:
            save(is_best=False, epoch=epoch)

        student_ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)

def test(epoch):
    student.eval()
    with torch.no_grad():
        if args.save_results:
            student_ckp.begin_background()
        
        student_ckp.write_log('\nEvaluation:')    
        student_ckp.add_log(torch.zeros(1, len(test_loader), len(args.scale)))
        
        timer_test = utility.timer()

        for idx_data, d in enumerate(test_loader):
            for idx_scale, scale in enumerate(args.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename, _ in tqdm(d, ncols=80):
                    lr, hr = prepare(lr, hr)
                    fms, sr = student(lr)
                    sr = utility.quantize(sr, args.rgb_range)
                    
                    save_list = [sr]
                    student_ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(sr, hr, scale, args.rgb_range, dataset=d)
                    if args.save_gt:
                        save_list.extend([lr, hr])

                    if args.save_results:
                        student_ckp.save_results(d, filename[0], save_list, scale)

                student_ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = student_ckp.log.max(0)
                student_ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        student_ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )
        student_ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        student_ckp.write_log('Saving...')

        if args.save_results:
            student_ckp.end_background()

        save(is_best=(best[1][0, 0] + 1 == epoch), epoch=epoch)

        student_ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)


def save(is_best, epoch):
    save_root_path = student_ckp.dir
    
    # save model
    save_dirs = [os.path.join(save_root_path, 'model', 'model_latest.pt')]    
    if is_best:
        save_dirs.append(os.path.join(save_root_path, 'model', 'model_best.pt'))        
    if args.save_models:
        save_dirs.append(os.path.join(save_root_path, 'model', 'model_{}.pt'.format(epoch)))    
    for s in save_dirs:
        torch.save(student.state_dict(), s)
    
    # save loss
    criterion.save(save_root_path)
    criterion.plot_loss(save_root_path, epoch)
    
    # save optimizer
    optimizer.save(save_root_path)
    
    # save psnr
    # student_ckp.plot_psnr(epoch)
    torch.save(student_ckp.log, os.path.join(save_root_path, 'psnr_log.pt'))



def print_args():
    msg = ""
    msg += "Model settings\n"
    msg += "Teachers: %s\n" % args.teacher
    msg += "Student: %s\n" % args.model

    msg += "\n"
    
    msg += "Data Settings\n"
    msg += "RGB range: %d\n" % args.rgb_range
    msg += "Scale: %d\n" % args.scale
    size = args.patch_size * args.scale
    msg += "Input Image Size: (%d, %d, 3)\n" % (args.patch_size, args.patch_size)
    msg += "Output Image Size: (%d, %d, 3)\n" % (size, size)
    msg += "\n"
    
    msg += "Training Settings\n"
    msg += "Epochs: %d\n" % args.epochs
    msg += "Learning rate: %f\n" % args.lr
    msg += "Learning rate decay: %s\n" % args.decay
    msg += "\n"
    
    msg += "Distillation Settings\n"
    if args.alpha == 0 and args.feature_loss_used == 0:
        msg += "No distilation\n"
    else:
        msg += "Distillation type: \n"
        if args.alpha != 0:
            msg += "\tteacher supervision\n"
        if args.feature_loss_used == 1:
            msg += "\tfeature distillation\n"
            msg += "\t\ttype: %s\n" % args.feature_distilation_type.split("*")[1]
            msg += "\t\tposition: %s\n" % args.features
        
    msg += "\n\n"    
    
    return msg


def calculate_psnr(LR, HR, scale):
    crop = 6 + scale
    ans = 0
    for i in range(len(LR)):
        img1 = LR[i].clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()
        img2 = HR[i].clamp(0, 255).round().cpu().byte().permute(1, 2, 0).numpy()

        img1 = img1[crop:-crop, crop:-crop, :]
        img2 = img2[crop:-crop, crop:-crop, :]

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        mse = np.mean((img1 - img2)**2)
        if mse == 0:
            return float("inf")
        ans += 20 * np.log10(255.0 / np.sqrt(mse))
    return ans / len(LR)


if __name__ == "__main__":
    msg = print_args()

    print("Preparing Data ====================================>")

    train_loader = MyDataloader.get_loader(args.dataset_root, batch_size=args.batch_size,
                                           input_size=args.input_size, scale=args.scale, patch_size=args.patch_size,
                                           valid_div=args.valid_div, mode="train", dataset_len=args.dataset_len)
    valid_loader = MyDataloader.get_loader(args.dataset_root, batch_size=args.batch_size,
                                           input_size=args.input_size, scale=args.scale, patch_size=args.patch_size,
                                           valid_div=args.valid_div, mode="valid", dataset_len=args.dataset_len)

    teachers = load_teachers()
    student_ckp, student = create_student_model()
    criterion = prepare_criterion()
    optimizer = prepare_optimizer()

    student_ckp.write_log(msg)

    epoch = 1
    if args.resume == 1:                
        epoch = len(student_ckp.log) + 1
    

    print("Start Training ======================================>") 
    while epoch < args.epochs + 1:
        print("epoch " + str(epoch))
        train(epoch)
        valid(epoch)
        # test(epoch)
        epoch += 1
      

# python -u train.py --ckp_dir overall_distilation/rcan/SA_x4/ --scale 4 --teacher [RCAN] --model RCAN --alpha 0.5 --feature_loss_used 1 --feature_distilation_type 10*SA --features [1,2,3] --chop --patch_size 32 --dataset_len 10000 > train-10000.txt 2>&1
