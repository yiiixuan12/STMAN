import torch
import numpy as np
import argparse
import time
import os
from pathlib import Path
import util
try:
    import matplotlib.pyplot as plt  # Optional; training does not require plotting.
except ModuleNotFoundError:
    plt = None
from engine import trainer

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:3',help='')
parser.add_argument('--data',type=str,default='data/METR-LA',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--in_dim',type=int,default=2,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=207,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=100,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--artifact_dir',type=str,default='',help='directory for pred.npy/true.npy artifacts')
parser.add_argument('--eval_checkpoint',type=str,default='',help='load a checkpoint and run streaming test evaluation only')
parser.add_argument('--eval_val_loss',type=str,default='',help='optional validation loss to print with eval-only results')
parser.add_argument('--stream_source',type=str,default='',help='raw npz/h5/csv source; build Graph WaveNet windows on the fly')
parser.add_argument('--stream_seq_length_x',type=int,default=12,help='history length for on-the-fly streaming windows')
parser.add_argument('--stream_train_ratio',type=float,default=0.6,help='streaming train split ratio')
parser.add_argument('--stream_val_ratio',type=float,default=0.2,help='streaming validation split ratio')
parser.add_argument('--stream_test_ratio',type=float,default=0.2,help='streaming test split ratio')

args = parser.parse_args()


def stream_graphwavenet_test_metrics(engine, dataloader, scaler, device, horizon_count):
    mae_sum = torch.zeros(horizon_count, dtype=torch.float64)
    mape_sum = torch.zeros(horizon_count, dtype=torch.float64)
    rmse_sum = torch.zeros(horizon_count, dtype=torch.float64)
    count = torch.zeros(horizon_count, dtype=torch.float64)

    engine.model.eval()
    for x, y in dataloader["test_loader"].get_iterator():
        testx = torch.as_tensor(x, dtype=torch.float32, device=device).transpose(1, 3)
        real = torch.as_tensor(y, dtype=torch.float32, device=device).transpose(1, 3)[:, 0, :, :]
        with torch.no_grad():
            pred = engine.model(testx).transpose(1, 3)[:, 0, :, :]
        batch_size = min(pred.size(0), real.size(0))
        horizons = min(horizon_count, pred.size(-1), real.size(-1))
        pred = scaler.inverse_transform(pred[:batch_size, :, :horizons])
        real = real[:batch_size, :, :horizons]
        valid = real != 0
        err = pred - real
        zero = torch.zeros_like(err)
        abs_err = torch.where(valid, torch.abs(err), zero)
        sq_err = torch.where(valid, err * err, zero)
        safe_real = torch.where(valid, real, torch.ones_like(real))
        pct_err = torch.where(valid, torch.abs(err) / safe_real, zero)

        mae_sum[:horizons] += abs_err.sum(dim=(0, 1)).double().cpu()
        mape_sum[:horizons] += pct_err.sum(dim=(0, 1)).double().cpu()
        rmse_sum[:horizons] += sq_err.sum(dim=(0, 1)).double().cpu()
        count[:horizons] += valid.sum(dim=(0, 1)).double().cpu()

    safe_count = torch.clamp(count, min=1.0)
    mae = mae_sum / safe_count
    mape = mape_sum / safe_count
    rmse = torch.sqrt(rmse_sum / safe_count)
    return [(mae[i].item(), mape[i].item(), rmse[i].item()) for i in range(horizon_count)]


def print_graphwavenet_test_metrics(metrics_by_horizon):
    amae = []
    amape = []
    armse = []
    for i, metrics in enumerate(metrics_by_horizon):
        log = 'Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
        print(log.format(i+1, metrics[0], metrics[1], metrics[2]))
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])

    log = 'On average over {:d} horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
    print(log.format(len(metrics_by_horizon), np.mean(amae), np.mean(amape), np.mean(armse)))


def main():
    #set seed
    #torch.manual_seed(args.seed)
    #np.random.seed(args.seed)
    #load data
    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    if args.stream_source:
        print("using streaming raw-series windows...", args.stream_source, flush=True)
        dataloader = util.load_dataset_streaming(
            args.stream_source,
            seq_length_x=args.stream_seq_length_x,
            horizon=args.seq_length,
            batch_size=args.batch_size,
            valid_batch_size=args.batch_size,
            test_batch_size=args.batch_size,
            train_ratio=args.stream_train_ratio,
            val_ratio=args.stream_val_ratio,
            test_ratio=args.stream_test_ratio,
        )
    elif args.eval_checkpoint:
        dataloader = util.load_dataset_eval_only(args.data, args.batch_size)
    else:
        dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None



    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)

    if args.eval_checkpoint:
        print("loading eval checkpoint...", args.eval_checkpoint, flush=True)
        engine.model.load_state_dict(torch.load(args.eval_checkpoint, map_location=device))
        print("Training skipped; streaming evaluation only", flush=True)
        print("Training finished")
        if args.eval_val_loss:
            print("The valid loss on best model is", args.eval_val_loss)
        metrics_by_horizon = stream_graphwavenet_test_metrics(
            engine, dataloader, scaler, device, args.seq_length
        )
        print_graphwavenet_test_metrics(metrics_by_horizon)
        return


    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    for i in range(1,args.epochs+1):
        #if i % 10 == 0:
            #lr = max(0.000002,args.learning_rate * (0.1 ** (i // 10)))
            #for g in engine.optimizer.param_groups:
                #g['lr'] = lr
        train_loss = []
        train_mape = []
        train_rmse = []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0 :
                log = 'Iter: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                print(log.format(iter, train_loss[-1], train_mape[-1], train_rmse[-1]),flush=True)
        t2 = time.time()
        train_time.append(t2-t1)
        #validation
        valid_loss = []
        valid_mape = []
        valid_rmse = []


        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mape = np.mean(train_mape)
        mtrain_rmse = np.mean(train_rmse)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mape = np.mean(valid_mape)
        mvalid_rmse = np.mean(valid_rmse)
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}, Valid Loss: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mape, mvalid_rmse, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))

    #testing
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(args.save+"_epoch_"+str(bestid+1)+"_"+str(round(his_loss[bestid],2))+".pth"))


    print("Training finished")
    print("The valid loss on best model is", str(round(his_loss[bestid],4)))


    metrics_by_horizon = stream_graphwavenet_test_metrics(
        engine, dataloader, scaler, device, args.seq_length
    )
    print_graphwavenet_test_metrics(metrics_by_horizon)
    torch.save(engine.model.state_dict(), args.save+"_exp"+str(args.expid)+"_best_"+str(round(his_loss[bestid],2))+".pth")
    if args.artifact_dir:
        print("Skipped prediction artifacts because streaming evaluation is enabled.", flush=True)



if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
