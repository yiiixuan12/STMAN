import torch
import torch.nn as nn
from model.DEGCRN import DEGCRN


class PM(nn.Module):
    def __init__(self,
                 node_num,   
                 dim_in,   
                 dim_out,   
                 cheb_k,   
                 embed_dim): 
        super(PM, self).__init__()

        self.node_num = node_num
        self.input_dim = dim_in

        self.stfgrn_cells = DEGCRN(node_num, dim_in, dim_out, cheb_k, embed_dim)

    def forward(self,
                x,   
                init_state,   
                node_embeddings):  
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1] 
        current_inputs = x  
        output_hidden = []
        state = init_state  
        inner_states = []
        for t in range(seq_length): 
            state = self.stfgrn_cells(current_inputs[:, t, :, :], state, [node_embeddings[0][:, t, :, :], node_embeddings[1]])
            inner_states.append(state)
        output_hidden.append(state)  
        current_inputs = torch.stack(inner_states, dim=1)   
        return current_inputs, output_hidden

    def init_hidden(self,
                    batch_size):  
        init_states = self.stfgrn_cells.init_hidden_state(batch_size) 
        return init_states 

class MFGCRM(nn.Module):
    def __init__(self,
                 node_num,   
                 dim_in,   
                 dim_out,   
                 cheb_k,  
                 embed_dim): 
        super(MFGCRM, self).__init__()

        self.node_num = node_num
        self.input_dim = dim_in

        self.dim_out = dim_out
        self.STFGRNS = nn.ModuleList()

        self.STFGRNS.append(PM(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(2):
            self.STFGRNS.append(PM(node_num, dim_in, dim_out, cheb_k, embed_dim))

    def forward(self,
                x,   
                node_embeddings): 
        init_state_R = self.STFGRNS[0].init_hidden(x.shape[0])  # (64,170,64)
        init_state_L = self.STFGRNS[1].init_hidden(x.shape[0]) 

        
        h_out = torch.zeros(x.shape[0], x.shape[1], x.shape[2],  self.dim_out* 2).to(x.device)  
        out1 = self.STFGRNS[0](x, init_state_R, node_embeddings)[0]  
        out2 = self.STFGRNS[1](torch.flip(x, [1]), init_state_L, node_embeddings)[0] 

        h_out[:, :, :, :self.dim_out] = out1
        h_out[:, :, :, self.dim_out:] = out2
        return h_out 

class DMFGCRN(nn.Module):
    def __init__(self, args):
        super(DMFGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.lag = args.lag
        self.num_layers = args.num_layers
        self.use_D = args.use_day
        self.use_W = args.use_week
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.default_graph = args.default_graph
        self.node_embeddings1 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.node_embeddings2 = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.T_i_D_emb = nn.Parameter(torch.empty(288, args.embed_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, args.embed_dim))

        self.encoder1 = MFGCRM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                              args.embed_dim)
        self.encoder2 = MFGCRM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                              args.embed_dim)

        ############---3
        self.dropout3 = nn.Dropout(p=0.1)
        self.encoder3 = MFGCRM(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                                 args.embed_dim)
        self.end_conv4 = nn.Conv2d(1, self.lag * self.output_dim, kernel_size=(1, 2 * self.hidden_dim), bias=True)
        self.end_conv5 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, 2 * self.hidden_dim), bias=True)

        #predictor
        self.end_conv1 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, 2*self.hidden_dim), bias=True)
        self.end_conv2 = nn.Conv2d(1, self.lag * self.output_dim, kernel_size=(1, 2*self.hidden_dim), bias=True)
        self.end_conv3 = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, 2*self.hidden_dim), bias=True)


    def forward(self, source, i=2):
        node_embedding1 = self.node_embeddings1
        if self.use_D:
            t_i_d_data   = source[..., 1]
            T_i_D_emb = self.T_i_D_emb[(t_i_d_data * 288).type(torch.LongTensor)]
            node_embedding1 = torch.mul(node_embedding1, T_i_D_emb)

        if self.use_W:
            d_i_w_data   = source[..., 2]
            D_i_W_emb = self.D_i_W_emb[(d_i_w_data).type(torch.LongTensor)]
            node_embedding1 = torch.mul(node_embedding1, D_i_W_emb)

        node_embeddings=[node_embedding1,self.node_embeddings1]

        source = source[..., 0].unsqueeze(-1)

        if i == 1:
            init_state1 = self.encoder1.init_hidden(source.shape[0])  
            output, _ = self.encoder1(source, init_state1, node_embeddings) 
            output = self.dropout1(output[:, -1:, :, :])
            output1 = self.end_conv1(output)  

            return output1

        else:
            output = self.encoder1(source, node_embeddings)
            output = self.dropout1(output[:, -1:, :, :])

          
            source1 = self.end_conv2(output)  
            output = self.end_conv1(output)    



            source1 = source -source1 

          
            output1 = self.encoder2(source1, node_embeddings)
                               
            output1 = self.dropout2(output1[:, -1:, :, :])

         
            source2 = self.end_conv4(output1) 
            output1 = self.end_conv3(output1) 

          

            source3 = source1- source2 
            output2 = self.encoder3(source3, node_embeddings)
            output2 = self.dropout3(output2[:, -1:, :, :])
            output2 = self.end_conv5(output2) 

            return output + output1 + output2
