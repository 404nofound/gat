# GAT���Ķ��壺layers.py

import numpy as np
import tensorflow as tf

conv1d = tf.layers.conv1d

def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):

    # seq ָ��������Ľڵ��������󣬴�СΪ [num_graph, num_node, fea_size]
    # out_sz ָ���Ǳ任��Ľڵ�����ά�ȣ�Ҳ���� W h_i��Ľڵ��ʾά��
    # bias_mat �Ǿ����任����ڽӾ��󣬴�СΪ [num_node, num_node]

    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        '''
        �������ȶ�ԭʼ�ڵ����� seq ���þ���˴�СΪ 1 �� 1D ���ģ��ͶӰ�任�õ��� seq_fts��
        ͶӰ�任���ά��Ϊ out_sz��ע�⣬����ͶӰ���� W �����нڵ㹲������ 1D ����е�
        ��������Ҳ�ǹ���ġ�

        ���seq_fts ��Ӧ�ڹ�ʽ�е� W h��shapeΪ [num_graph, num_node, out_sz]��

        '''
        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False) # [num_graph, num_node, out_sz]

        # simplest self-attention possible
        '''
        ͶӰ�任��õ���seq_fts����ʹ�þ���˴�СΪ 1 �� 1D ��������õ��ڵ㱾���ͶӰf_1 �� ���ھӵ�ͶӰf_2��
        ��Ӧ�����Ĺ�ʽ�е�a(Wh_i, Wh_j)��ע����������ͶӰ�Ĳ����Ƿֿ��ģ���������ͶӰ����a_1��a_2��
        �ֱ��Ӧ��������conv1d �еĲ�����

        ���� tf.layers.conv1d(seq_fts, 1, 1) ֮��� f_1 �� f_2 ��Ӧ�ڹ�ʽ�е� a_1Wh_i �� a_2Wh_j��
        ά�Ⱦ�Ϊ [num_graph, num_node, 1]��

        '''
        f_1 = tf.layers.conv1d(seq_fts, 1, 1) # [num_graph, num_node, 1]
        f_2 = tf.layers.conv1d(seq_fts, 1, 1) # [num_graph, num_node, 1]

        # �� f_2 ת��֮���� f_1 ���ӣ�ͨ��Tensorflow�Ĺ㲥���Ƶõ��Ĵ�СΪ [num_graph, num_node, num_node] ��
        # logits������һ��ע��������
        logits = f_1 + tf.transpose(f_2, [0, 2, 1]) # [num_graph, num_node, num_node]

        # ������, ���� GAT �Ĺ�ʽ������ֻҪ�� logits ���� softmax ��һ���Ϳ��Եõ�ע����Ȩ�� ��Ҳ���Ǵ������ coefs��
        '''
        ���ǣ�����Ϊʲô���һ�� bias_mat �أ�

        ��Ϊ�� logits �洢�����������ڵ�֮���ע����ֵ�����ǣ���һ��ֻ��Ҫ��ÿ���ڵ�������ھӵ�ע��������
        ��k����N_i�������ԣ������� bias_mat ���ǽ� softmax �Ĺ�һ������Լ����ÿ���ڵ���ھ��ϡ�

        '''
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # ��󣬽� mask ֮���ע�������� coefs ��任����������� seq_fts ��ˣ����ɵõ����º�Ľڵ��ʾ vals
        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        
        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat*f_1
        f_2 = adj_mat * tf.transpose(f_2, [1,0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices, 
                values=tf.nn.leaky_relu(logits.values), 
                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1) # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

