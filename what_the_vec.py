import numpy as np
from load_data import Data
import torch
from torch.optim.lr_scheduler import ExponentialLR
import time
import argparse

class P2VL(torch.nn.Module):
    def __init__(self, dim):
        super(P2VL, self).__init__()
        self.W = torch.nn.Embedding(len(d.word_counts), dim, padding_idx=0)
        self.C = torch.nn.Embedding(len(d.word_counts), dim, padding_idx=0)
        self.loss = torch.nn.MSELoss()

    def forward(self, w_idx, c_idx):
        self.w_batch = self.W(w_idx)
        self.c_batch = self.C(c_idx)
        score = torch.sum(self.w_batch*self.c_batch, dim=1)
        score_w = torch.sqrt(torch.sum(self.w_batch*self.w_batch, dim=1))
        score_c = torch.sqrt(torch.sum(self.c_batch*self.c_batch, dim=1))
        return score, score_w, score_c

class P2VP(torch.nn.Module):
    def __init__(self, dim):
        super(P2VP, self).__init__()
        self.W = torch.nn.Embedding(len(d.word_counts), dim, padding_idx=0)
        self.C = torch.nn.Embedding(len(d.word_counts), dim, padding_idx=0)
        self.loss = torch.nn.MSELoss()

    def forward(self, w_idx, c_idx):
        self.w_batch = self.W(w_idx)
        self.c_batch = self.C(c_idx)
        self.c_w_batch = self.C(w_idx)
        score = torch.sum(self.w_batch*self.c_batch, dim=1)
        score_w = torch.sum(self.w_batch*self.c_w_batch, dim=1)
        score_c = torch.sqrt(torch.sum(self.w_batch*self.w_batch, dim=1)) -\
                  torch.sqrt(torch.sum(self.c_w_batch*self.c_w_batch, dim=1))
        return score, score_w, score_c


class Experiment:

    def __init__(self, model_name, learning_rate=0.1, embeddings_dim=200, 
                num_iterations=100, decay_rate=0.98, batch_size=10000, 
                corrupt_size=5, w_reg=0., c_reg=0., cuda=False):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.embeddings_dim = embeddings_dim
        self.num_iterations = num_iterations
        self.decay_rate = decay_rate
        self.batch_size = batch_size
        self.corrupt_size = corrupt_size
        self.w_reg = w_reg
        self.c_reg = c_reg
        self.cuda = cuda

    def train_and_eval(self):
        print("Training the %s model..." % self.model_name)

        outfolder = "/afs/inf.ed.ac.uk/group/project/Knowledge_Bases/stats/"
        fname = "result_%dmil_%s_ws%d_w%d_c%d_d%d_" % (int(np.ceil(d.cutoff/1e6)), 
                            self.model_name, d.window_size, int(self.w_reg*10), 
                            int(self.c_reg*10), self.embeddings_dim)

        if self.model_name.lower() == "p2v-l":
            model = P2VL(self.embeddings_dim)
        elif self.model_name.lower() == "p2v-p":
            model = P2VP(self.embeddings_dim)

        if self.cuda:
            model.cuda()

        opt = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        if self.decay_rate:
            scheduler = ExponentialLR(opt, self.decay_rate)

        data_idxs = range(len(d.word_counts))
        data_w_probs = [d.w_probs[i] for i in data_idxs]
        data_c_probs = [d.c_probs[i] for i in data_idxs]
        cooccurrences = list(d.cooccurrence_counts.keys())
        
        def pmi_ii(data_batch, i):
            return np.maximum(0.1, np.log([d.cooccurrence_counts.get((pair[i], pair[i]), min_p)/
                                           d.w_probs[pair[i]]**2 for pair in data_batch]))

        def pmi(data_batch):
            targets = [d.cooccurrence_counts.get(pair, 0.)/(d.w_probs[pair[0]]*
                       d.c_probs[pair[1]]) for pair in data_batch]
            return np.array([np.log(target) if target!=0 else -1. for target in targets])


        ww_cooccurrences = [d.cooccurrence_counts.get((idx, idx), 100.) for idx in d.w_probs]
        min_p = np.min(ww_cooccurrences)/1.5

        losses = []
        tick = time.time()
        counter = 0
        for i in range(1, self.num_iterations+1):
            model.train() 
            np.random.shuffle(cooccurrences)
            num_batches = int(np.ceil(len(cooccurrences)/float(self.batch_size)))
            num_neg_samples = self.batch_size*self.corrupt_size*num_batches
            all_neg_pairs = list(zip(np.random.choice(data_idxs, num_neg_samples, p=data_w_probs), 
                                     np.random.choice(data_idxs, num_neg_samples, p=data_c_probs)))
            
            epoch_loss = []
            for j in range(0, len(cooccurrences), self.batch_size):
                counter += 1
                pos_pairs = cooccurrences[j:min(j+self.batch_size, len(cooccurrences))]
                neg_pairs = all_neg_pairs[j*self.corrupt_size:j*self.corrupt_size+
                                          self.batch_size*self.corrupt_size]
                data_batch = pos_pairs + neg_pairs
                targets = torch.FloatTensor(pmi(data_batch))
                if self.model_name.lower() == "p2v-l":
                    targets_w = torch.FloatTensor(np.sqrt(pmi_ii(data_batch, 0)))
                    targets_c = torch.FloatTensor(np.sqrt(pmi_ii(data_batch, 1)))
                elif self.model_name.lower() == "p2v-p":
                    targets_w = torch.FloatTensor(pmi_ii(data_batch, 0))
                    targets_c = torch.FloatTensor(np.zeros(len(targets)))
                opt.zero_grad()
                data_batch = np.array(data_batch)
                w_idx = torch.tensor(data_batch[:,0])
                c_idx = torch.tensor(data_batch[:,1])
                if self.cuda:
                    w_idx = w_idx.cuda()
                    c_idx = c_idx.cuda()
                    targets = targets.cuda()
                    targets_w = targets_w.cuda()
                    targets_c = targets_c.cuda()
                preds, preds_w, preds_c = model.forward(w_idx, c_idx)
                loss = model.loss(preds, targets) +\
                       self.w_reg * model.loss(preds_w, targets_w) +\
                       self.c_reg * model.loss(preds_c, targets_c)
                loss.backward()
                opt.step()
                epoch_loss.append(loss.item())
                if self.decay_rate and not counter%500:
                    scheduler.step()
            print("Iteration: %d" % i)
            print("Loss: %.4f" % np.mean(epoch_loss))
            if not i%10:
                np_W = model.W.weight.detach().cpu().numpy()
                np_C = model.C.weight.detach().cpu().numpy()   
                if i == 10:
                    np.save("%s%s%d.npy" % (outfolder, fname, i), {"W":np_W, "C":np_C, 
                            "p_w":d.w_probs, "p_c":d.c_probs, "p_wc":d.cooccurrence_counts, 
                            "losses":losses, "i2w":d.idx_to_word})
                else:
                    np.save("%s%s%d.npy" % (outfolder, fname, i), {"W":np_W, "C":np_C, 
                            "losses":losses})
            losses.append(np.mean(epoch_loss))
        print("Time: ", str(time.time()-tick))


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available:
        torch.cuda.manual_seed_all(seed) 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="text8", nargs="?",
                    help='Which dataset to use')
    parser.add_argument('--window_size', type=int, default=10, nargs="?",
                    help='Window size')
    parser.add_argument('--min_occurrences', type=int, default=5, nargs="?",
                    help='Minimum number of occurrences of a word to include it in the stats')
    parser.add_argument('--subsample', type=bool, default=True, nargs="?",
                    help='Whether to randomly subsample frequent words or not')
    parser.add_argument('--threshold', type=float, default=1e-3, nargs="?",
                    help='Threshold for random subsampling')
    parser.add_argument('--cutoff', type=int, default=None, nargs="?",
                    help='Cutoff of the dataset (if you want to test it on smaller data' \
                          'set it to none otherwise)')
    parser.add_argument('--model', type=str, default="p2v-l", nargs="?",
                    help='Which model to use: p2v-l or p2v-p')
    parser.add_argument('--num_iters', type=int, default=100, nargs="?",
                    help='Number of iterations')
    parser.add_argument('--lr', type=float, default=0.1, nargs="?",
                    help='Initial learning rate')
    parser.add_argument('--dr', type=float, default=0.98, nargs="?",
                    help='Decay rate')
    parser.add_argument('--batch_size', type=int, default=10000, nargs="?",
                    help='Batch size')
    parser.add_argument('--num_neg', type=int, default=5, nargs="?",
                    help='Number of negative samples per each positive sample')
    parser.add_argument('--dim', type=int, default=200, nargs="?",
                    help='Embeddings dimensionality')
    parser.add_argument('--w_reg', type=float, default=0.5, nargs="?",
                    help='Regularization coefficient for W')
    parser.add_argument('--c_reg', type=float, default=0.5, nargs="?",
                    help='Regularization coefficient for C')
    parser.add_argument('--cuda', type=bool, default=True, nargs="?",
                    help='Whether to use cuda (GPU) or not (CPU)')

    args = parser.parse_args()
    d = Data(data_dir="data/", fname=args.dataset, min_occurrences=args.min_occurrences, 
             window_size=args.window_size, subsample=args.subsample, t=args.threshold,
             cutoff=args.cutoff)
    
    experiment = Experiment(args.model, num_iterations=args.num_iters, learning_rate=args.lr, 
                    batch_size=args.batch_size, corrupt_size=args.num_neg, decay_rate=args.dr, 
                    embeddings_dim=args.dim, w_reg=args.w_reg, c_reg=args.c_reg, cuda=args.cuda)
    experiment.train_and_eval()
    