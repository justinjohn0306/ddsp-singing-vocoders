import os
import sys
import time
import shutil
import regex as re
import numpy as np
from scipy import signal
import soundfile as sf
import parselmouth

import torch

from logger.saver import Saver
from logger import utils

def apodize(values, minidx, maxidx, length):
    values[minidx-length:minidx] *= np.linspace(1.0,0.0,length)
    values[minidx:maxidx] = 0.0
    values[maxidx:maxidx+length] *= np.linspace(0.0,1.0,length)

def apodize2(values, minidx, maxidx, length, sr):
    sos = signal.butter(N=5, Wn=2000, fs=sr, output='sos')
    filtered_range = signal.sosfiltfilt(sos,
        values[minidx-length:maxidx+length])
    filtered_range[0:length] *= np.linspace(0.0,1.0,length)
    filtered_range[-length:] *= np.linspace(1.0,0.0,length)

    values[minidx-length:minidx] *= np.linspace(1.0,0.0,length)
    values[minidx:maxidx] = 0.0
    values[maxidx:maxidx+length] *= np.linspace(0.0,1.0,length)

    values[minidx-length:maxidx+length] += filtered_range

def render(args, model, path_mel_dir, path_gendir='gen', is_part=False, debuzz=False):
    print(' [*] rendering...')
    model.eval()

    # list files
    files = utils.traverse_dir(
        path_mel_dir, 
        extension='npy', 
        is_ext=False,
        is_sort=True, 
        is_pure=True)
    num_files = len(files)
    print(' > num_files:', num_files)

    # run
    with torch.no_grad():
        for fidx in range(num_files):
            fn = files[fidx]
            print('--------')
            print('{}/{} - {}'.format(fidx, num_files, fn))

            path_mel = os.path.join(path_mel_dir, fn) + '.npy'
            mel = np.load(path_mel)
            mel = torch.from_numpy(mel).float().to(args.device).unsqueeze(0)
            print(' mel:', mel.shape)

            # forward
            signal, f0_pred, _, (s_h, s_n) = model(mel)

            # path
            path_pred = os.path.join(path_gendir, 'pred', fn + '.wav')
            if is_part:
                path_pred_n = os.path.join(path_gendir, 'part', fn + '-noise.wav')
                path_pred_h = os.path.join(path_gendir, 'part', fn + '-harmonic.wav')
            print(' > path_pred:', path_pred)
            
            os.makedirs(os.path.dirname(path_pred), exist_ok=True)
            if is_part:
                os.makedirs(os.path.dirname(path_pred_h), exist_ok=True)

            # to numpy
            pred = utils.convert_tensor_to_numpy(signal)
            pred_n = utils.convert_tensor_to_numpy(s_n)
            pred_h = utils.convert_tensor_to_numpy(s_h)

            if debuzz:
                # load wave
                wave_pred = parselmouth.Sound(pred.astype(np.float64),
                    sampling_frequency=args.data.sampling_rate)
                wave_noise = parselmouth.Sound(pred_n.astype(np.float64),
                    sampling_frequency=args.data.sampling_rate)
                wave_harmonic = parselmouth.Sound(pred_h.astype(np.float64),
                    sampling_frequency=args.data.sampling_rate)

                # extract UV
                pitch = wave_pred.to_pitch_ac(
                    args.debuzz.pitch_time_step, 
                    args.debuzz.pitch_floor,
                    args.debuzz.max_candidates,
                    args.debuzz.very_accurate,
                    args.debuzz.silence_thresh,
                    args.debuzz.voicing_thresh,
                    args.debuzz.octave_cost,
                    args.debuzz.oct_jump_cost,
                    args.debuzz.vuv_cost,
                    args.debuzz.pitch_ceiling)
                pitch_values = pitch.selected_array['frequency']
                pitch_values[pitch_values==0] = np.nan
                UV_Indices = np.argwhere(np.isnan(pitch_values)).flatten()

                # apply UV on harmonic parts
                step = int(args.debuzz.pitch_time_step/2 *
                        wave_harmonic.sampling_frequency) + 1
                for index in UV_Indices:
                    # upsample f0 to sample level
                    h_index = (np.abs(wave_harmonic.xs() -
                        pitch.xs()[index])).argmin() 
                    #apodize(wave_harmonic.values[0], h_index-step,
                        #h_index+step, length=args.debuzz.fade_length)
                    apodize2(wave_harmonic.values[0], h_index-step,
                        h_index+step, length=args.debuzz.fade_length,
                        sr=wave_harmonic.sampling_frequency)

                # the first and last 0.25 seconds don't have pitch detection,
                # so mute these?

                # this is VERY evident in the final audio

                # trim = int(wave_harmonic.sampling_frequency * 0.25)+1
                # wave_harmonic.values[0][:trim] = 0
                # wave_harmonic.values[0][-trim:] = 0
                
                # combine two parts
                wave_final = wave_harmonic.values + wave_noise.values
                pred = np.squeeze(wave_final)
            
            # save
            sf.write(path_pred, pred, args.data.sampling_rate)
            if is_part:
                sf.write(path_pred_n, pred_n, args.data.sampling_rate)
                sf.write(path_pred_h, pred_h, args.data.sampling_rate)


def test(args, model, loss_func, loader_test, path_gendir='gen', is_part=False):
    print(' [*] testing...')
    print(' [*] output folder:', path_gendir)
    model.eval()

    # losses
    test_loss = 0.
    test_loss_mss = 0.
    test_loss_f0 = 0.
    
    # intialization
    num_batches = len(loader_test)
    os.makedirs(path_gendir, exist_ok=True)
    rtf_all = []

    # run
    with torch.no_grad():
        for bidx, data in enumerate(loader_test):
            fn = data['name'][0]
            print('--------')
            print('{}/{} - {}'.format(bidx, num_batches, fn))

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device).float()
            print('>>', data['name'][0])

            # forward
            st_time = time.time()
            signal, f0_pred, _, (s_h, s_n) = model(data['mel'])
            ed_time = time.time()

            # crop
            min_len = np.min([signal.shape[1], data['audio'].shape[1]])
            signal        = signal[:,:min_len]
            data['audio'] = data['audio'][:,:min_len]

            # RTF
            run_time = ed_time - st_time
            song_time = data['audio'].shape[-1] / args.data.sampling_rate
            rtf = run_time / song_time
            print('RTF: {}  | {} / {}'.format(rtf, run_time, song_time))
            rtf_all.append(rtf)
           
            # loss
            loss, (loss_mss, loss_f0) = loss_func(
                signal, data['audio'], f0_pred, data['f0'])

            test_loss         += loss.item()
            test_loss_mss     += loss_mss.item() 
            test_loss_f0      += loss_f0.item()

            # path
            path_pred = os.path.join(path_gendir, 'pred', fn + '.wav')
            path_anno = os.path.join(path_gendir, 'anno', fn + '.wav')
            if is_part:
                path_pred_n = os.path.join(path_gendir, 'part', fn + '-noise.wav')
                path_pred_h = os.path.join(path_gendir, 'part', fn + '-harmonic.wav')

            print(' > path_pred:', path_pred)
            print(' > path_anno:', path_anno)

            os.makedirs(os.path.dirname(path_pred), exist_ok=True)
            os.makedirs(os.path.dirname(path_anno), exist_ok=True)
            if is_part:
                os.makedirs(os.path.dirname(path_pred_h), exist_ok=True)

            # to numpy
            pred  = utils.convert_tensor_to_numpy(signal)
            anno  = utils.convert_tensor_to_numpy(data['audio'])
            if is_part:
                pred_n = utils.convert_tensor_to_numpy(s_n)
                pred_h = utils.convert_tensor_to_numpy(s_h)
            
            # save
            sf.write(path_pred, pred, args.data.sampling_rate)
            sf.write(path_anno, anno, args.data.sampling_rate)
            if is_part:
                sf.write(path_pred_n, pred_n, args.data.sampling_rate)
                sf.write(path_pred_h, pred_h, args.data.sampling_rate)
            
    # report
    test_loss /= num_batches
    test_loss_mss     /= num_batches
    test_loss_f0      /= num_batches

    # check
    print(' [test_loss] test_loss:', test_loss)
    print(' Real Time Factor', np.mean(rtf_all))
    return test_loss, test_loss_mss, test_loss_f0

def warmstart_model(args, saver, model):
    if os.path.exists(saver.expdir) is None: return
    if os.path.exists(os.path.join(saver.expdir, "ckpts")) is None: return
    print("Checkpoints exist, trying warmstart")

    max_time = 0
    max_ckpt_name = ""
    max_ckpt_num = 0

    for root, _, files in os.walk(os.path.join(saver.expdir, "ckpts")):
        for name in files:
            match = re.match('vocoder_(\d+)_\d+\.\d+_best_params.pt', name)
            if match:
                max_ckpt_num = int(match.group(1))
                max_ckpt_name = name

                
    if len(max_ckpt_name) != 0:
        print("Using warmstart checkpoint "+max_ckpt_name)
        utils.load_model_params(os.path.join(saver.expdir, "ckpts",
            max_ckpt_name), model, args.device)

    return model, max_ckpt_num

def train(args, model, loss_func, loader_train, loader_test):
    # saver
    saver = Saver(args)

    # model size
    params_count = utils.get_network_paras_amount({'model': model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.lr)

    # warmstart if available
    model, max_ckpt_num = warmstart_model(args, saver, model)
    saver.warmstart_step(max_ckpt_num)
    loss_func.update_f0_iter(max_ckpt_num)

    # run
    best_loss = np.inf
    num_batches = len(loader_train)
    model.train()
    prev_save_time = -1
    saver.log_info('======= start training =======')
    for epoch in range(args.train.epochs):

        for batch_idx, data in enumerate(loader_train):
            saver.global_step_increment()
            optimizer.zero_grad()

            # unpack data
            for k in data.keys():
                if k != 'name':
                    data[k] = data[k].to(args.device).float()
            
            # forward
            signal, f0_pred, _, _,  = model(data['mel'])

            # loss
            loss, (loss_mss, loss_f0) = loss_func(
                signal, data['audio'], f0_pred, data['f0'])
            
            # handle nan loss
            if torch.isnan(loss):
                raise ValueError(' [x] nan loss ')
            else:
                # backpropagate
                loss.backward()
                optimizer.step()

            # log loss
            if saver.global_step % args.train.interval_log == 0:
                saver.log_info(
                    'epoch: {}/{} {:3d}/{:3d} | {} | t: {:.2f} | loss: {:.6f} | time: {} | counter: {}'.format(
                        epoch,
                        args.train.epochs,
                        batch_idx,
                        num_batches,
                        saver.expdir,
                        saver.get_interval_time(),
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
                saver.log_info(
                    ' > mss loss: {:.6f}, f0: {:.6f}'.format(
                       loss_mss.item(),
                       loss_f0.item(),
                    )
                )

                y, s = signal, data['audio']
                saver.log_info(
                    "pred: max:{:.5f}, min:{:.5f}, mean:{:.5f}, rms: {:.5f}\n" \
                    "anno: max:{:.5f}, min:{:.5f}, mean:{:.5f}, rms: {:.5f}".format(
                            torch.max(y), torch.min(y), torch.mean(y), torch.mean(y** 2) ** 0.5,
                            torch.max(s), torch.min(s), torch.mean(s), torch.mean(s** 2) ** 0.5))

                saver.log_value({
                    'train loss': loss.item(), 
                    'train loss mss': loss_mss.item(),
                    'train loss f0': loss_f0.item(),
                })
            
            # validation
            # if saver.global_step % args.train.interval_val == 0:
            cur_hour = saver.get_total_time(to_str=False) // 1800
            if cur_hour != prev_save_time:
                # save latest
                saver.save_models(
                        {'vocoder': model}, postfix=f'{saver.global_step}_{cur_hour}')

                prev_save_time = cur_hour
                # run testing set
                path_testdir_runtime = os.path.join(
                        args.env.expdir,
                        'runtime_gen', 
                        f'gen_{saver.global_step}_{cur_hour}')
                test_loss, test_loss_mss, test_loss_f0 = test(
                    args, model, loss_func, loader_test,
                    path_gendir=path_testdir_runtime)
                saver.log_info(
                    ' --- <validation> --- \nloss: {:.6f}. mss loss: {:.6f}, f0: {:.6f}'.format(
                        test_loss, test_loss_mss, test_loss_f0
                    )
                )

                saver.log_value({
                    'valid loss': test_loss,
                    'valid loss mss': test_loss_mss,
                    'valid loss f0': test_loss_f0,
                })
                model.train()

                # save best model
                if test_loss < best_loss:
                    saver.log_info(' [V] best model updated.')
                    saver.save_models(
                        {'vocoder': model}, postfix=
                        f'{saver.global_step}_{cur_hour}_best')
                    # That's a pretty big mistake to make
                    # test_loss = best_loss
                    best_loss = test_loss

                saver.make_report()

                          
