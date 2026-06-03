from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric

import csv
import json
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time

import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm

class Exp_Informer(Exp_Basic):
    def __init__(self, config):
        self.config = config  # 先设置config属性
        super(Exp_Informer, self).__init__(config)  # 再调用父类初始化
    
    def _build_model(self):
        model = Informer(
            self.config.enc_in,
            self.config.dec_in, 
            self.config.c_out, 
            self.config.seq_len, 
            self.config.label_len,
            self.config.pred_len,
            self.config.factor,
            self.config.d_model, 
            self.config.n_heads, 
            self.config.e_layers,
            self.config.d_layers, 
            self.config.d_ff,
            self.config.dropout, 
            self.config.attn,
            self.config.embed,
            self.config.freq,
            self.config.activation,
            self.config.output_attention,
            self.config.distil,
            self.config.mix,
            device=self.device,
            encoder_lee_types=self.config.encoder_lee_types,
            decoder_lee_types=self.config.decoder_lee_types,
            perturb_amplitude=self.config.perturb_amplitude,
            perturb_frequency=self.config.perturb_frequency,
            perturb_phase=self.config.perturb_phase,
            lee_type=self.config.lee_type,
            encoder_activation=self.config.encoder_activation,
            decoder_activation=self.config.decoder_activation,
            output_activation=self.config.output_activation,
        ).to(self.device)
        
        return model

    def _get_data(self, flag):
        args = self.config

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'Solar1':Dataset_Custom,
            'Solar5':Dataset_Custom,
            'Weather':Dataset_Custom,
            'Exchange':Dataset_Custom,
            'ILI':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.config.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
                pred, true = self._process_one_batch(
                    vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        path = os.path.join(self.config.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.config.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.config.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model = self.model.to(self.device)
            self.model.train()
            epoch_time = time.time()
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.train_epochs}')
            
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_bar):
                iter_count += 1
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:,-self.config.pred_len:,:]).float()
                dec_inp = torch.cat([batch_y[:,:self.config.label_len,:], dec_inp], dim=1).float().to(self.device)

                if self.config.use_amp:
                    with torch.cuda.amp.autocast():
                        self._set_dynamic_activation(batch_x)
                        if self.config.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.config.features=='MS' else 0
                        batch_y = batch_y[:,-self.config.pred_len:,f_dim:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    self._set_dynamic_activation(batch_x)
                    if self.config.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.config.features=='MS' else 0
                    batch_y = batch_y[:,-self.config.pred_len:,f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.config.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.config.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                
                train_bar.set_postfix({'loss': '{:.6f}'.format(loss.item())})

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch+1, self.config)
            
        best_model_path = path+'/'+'checkpoint.pth'
        torch.save(self.model.state_dict(), best_model_path)
        
        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
                pred, true = self._process_one_batch(
                    test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred.detach().cpu().numpy())
                trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = os.path.join(self.config.results_dir, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(os.path.join(folder_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
        if self.config.save_pred:
            np.save(os.path.join(folder_path, 'pred.npy'), preds)
        if self.config.save_true:
            np.save(os.path.join(folder_path, 'true.npy'), trues)
        if self.config.save_config:
            self._save_config(folder_path, setting)
        self._append_summary(setting, mae, mse, rmse, mape, mspe)

        return mae, mse, rmse, mape, mspe

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.config.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        with torch.no_grad():
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
                pred, true = self._process_one_batch(
                    pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = os.path.join(self.config.results_dir, setting)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(os.path.join(folder_path, 'real_prediction.npy'), preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.config.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.config.pred_len, batch_y.shape[-1]]).float()
        elif self.config.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.config.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.config.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.config.use_amp:
            with torch.cuda.amp.autocast():
                self._set_dynamic_activation(batch_x)
                if self.config.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            self._set_dynamic_activation(batch_x)
            if self.config.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.config.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.config.features=='MS' else 0
        batch_y = batch_y[:,-self.config.pred_len:,f_dim:].to(self.device)

        return outputs, batch_y

    def _set_dynamic_activation(self, batch_x):
        if self.config.activation != 'dynamic_gelu_sin':
            return
        target_series = batch_x[:, :, -1]
        volatility = torch.std(target_series, dim=1, unbiased=False)
        mean_abs_change = torch.mean(torch.abs(target_series[:, 1:] - target_series[:, :-1]), dim=1)
        centered = target_series - torch.mean(target_series, dim=1, keepdim=True)
        scale = torch.std(target_series, dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
        anomaly_density = torch.mean((torch.abs(centered / scale) > 2.0).float(), dim=1)
        score = volatility + mean_abs_change + anomaly_density
        if score.numel() >= 3:
            low_cut = torch.quantile(score.detach(), 1.0 / 3.0)
            high_cut = torch.quantile(score.detach(), 2.0 / 3.0)
        else:
            low_cut = high_cut = torch.median(score.detach())
        low = torch.full_like(score, float(self.config.dynamic_low_amplitude))
        mid = torch.full_like(score, float(self.config.dynamic_mid_amplitude))
        high = torch.full_like(score, float(self.config.dynamic_high_amplitude))
        amplitudes = torch.where(score <= low_cut, low, torch.where(score >= high_cut, high, mid))
        for module in self.model.modules():
            if hasattr(module, 'set_dynamic_amplitude'):
                module.set_dynamic_amplitude(amplitudes.detach())

    def smoke_test_data(self, flag='train'):
        data_set, data_loader = self._get_data(flag=flag)
        batch = next(iter(data_loader))
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        result = {
            'dataset': self.config.data,
            'flag': flag,
            'root_path': self.config.root_path,
            'data_path': self.config.data_path,
            'target': self.config.target,
            'features': self.config.features,
            'num_samples': len(data_set),
            'batch_x_shape': list(batch_x.shape),
            'batch_y_shape': list(batch_y.shape),
            'batch_x_mark_shape': list(batch_x_mark.shape),
            'batch_y_mark_shape': list(batch_y_mark.shape),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    def _config_dict(self):
        config = {}
        for key, value in vars(self.config).items():
            if key.startswith('_'):
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                config[key] = value
            elif isinstance(value, (list, tuple)):
                config[key] = list(value)
            else:
                config[key] = str(value)
        return config

    def _save_config(self, folder_path, setting):
        payload = self._config_dict()
        payload['setting'] = setting
        with open(os.path.join(folder_path, 'config.json'), 'w') as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    def _append_summary(self, setting, mae, mse, rmse, mape, mspe):
        summary_path = self.config.summary_path
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        row = {
            'setting': setting,
            'dataset': self.config.data,
            'pred_len': self.config.pred_len,
            'activation': self.config.activation,
            'encoder_activation': self.config.encoder_activation,
            'decoder_activation': self.config.decoder_activation,
            'output_activation': self.config.output_activation,
            'activation_signature': self.config.activation_signature,
            'activation_family': self.config.activation_family,
            'amplitude': self.config.perturb_amplitude,
            'frequency': self.config.perturb_frequency,
            'phase': self.config.perturb_phase,
            'lee_type': self.config.lee_type,
            'features': self.config.features,
            'target': self.config.target,
            'seq_len': self.config.seq_len,
            'label_len': self.config.label_len,
            'd_model': self.config.d_model,
            'n_heads': self.config.n_heads,
            'e_layers': self.config.e_layers,
            'd_layers': self.config.d_layers,
            'd_ff': self.config.d_ff,
            'factor': self.config.factor,
            'dropout': self.config.dropout,
            'attn': self.config.attn,
            'embed': self.config.embed,
            'batch_size': self.config.batch_size,
            'train_epochs': self.config.train_epochs,
            'seed': self.config.seed,
            'server_id': self.config.server_id,
            'server_ip': self.config.server_ip,
            'des': self.config.des or '',
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'mspe': mspe,
            'relative_mse_change': '',
            'win_against_baseline': '',
            'run_id': self.config.run_id or '',
        }
        fieldnames = list(row.keys())
        write_header = not os.path.exists(summary_path)
        if not write_header:
            with open(summary_path, newline='') as f:
                reader = csv.DictReader(f)
                existing_fieldnames = reader.fieldnames or []
                existing_rows = list(reader)
            if existing_fieldnames != fieldnames:
                fieldnames = existing_fieldnames + [
                    field for field in fieldnames if field not in existing_fieldnames
                ]
                with open(summary_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for existing_row in existing_rows:
                        writer.writerow({
                            field: existing_row.get(field, '')
                            for field in fieldnames
                        })
        with open(summary_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def grid_search(self):
        """执行网格搜索以找到最优的Lee振荡器配置"""
        if not self.config.lee_grid_search:
            return self.train('default_setting')
            
        best_loss = float('inf')
        best_config = None
        lee_types = self.config.lee_types_to_try
        
        # 生成所有可能的组合
        from itertools import product
        encoder_combinations = list(product(lee_types, repeat=len(self.config.encoder_lee_types)))
        decoder_combinations = list(product(lee_types, repeat=len(self.config.decoder_lee_types)))
        
        results = []
        for enc_types in encoder_combinations:
            for dec_types in decoder_combinations:
                print(f"\n尝试配置: Encoder types {enc_types}, Decoder types {dec_types}")
                
                # 更新配置
                self.config.encoder_lee_types = list(enc_types)
                self.config.decoder_lee_types = list(dec_types)
                
                # 重建模型
                self.model = self._build_model()
                
                # 训练模型
                setting = f'lee_enc{"".join(map(str,enc_types))}_dec{"".join(map(str,dec_types))}'
                model = self.train(setting)
                
                # 评估模型
                _, val_loader = self._get_data(flag='val')
                val_loss = self.vali(None, val_loader, self._select_criterion())
                
                results.append({
                    'encoder_types': enc_types,
                    'decoder_types': dec_types,
                    'val_loss': val_loss,
                    'setting': setting
                })
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    best_config = {
                        'encoder_types': enc_types,
                        'decoder_types': dec_types,
                        'setting': setting
                    }
                
                print(f"验证损失: {val_loss:.6f}")
        
        # 保存所有结果
        import json
        with open('grid_search_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n最佳配置:")
        print(f"编码器Lee类型: {best_config['encoder_types']}")
        print(f"解码器Lee类型: {best_config['decoder_types']}")
        print(f"最佳验证损失: {best_loss:.6f}")
        
        return best_config
