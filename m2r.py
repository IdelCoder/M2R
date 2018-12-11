# encoding=utf-8
import numpy as np
import tensorflow as tf
import random
from sklearn.feature_extraction.text import CountVectorizer
from numpy import linalg as LA
import re
np.set_printoptions(threshold=np.nan)
input_dir = "./data/"
relation2id = input_dir + 'nyt_relation2id.txt'
# training dataset
f_train_m2id = input_dir + "train_mention2id.txt"
f_train_r2m = input_dir + "train_rmpair.txt"
# testing dataset
f_test_m2id = input_dir + "dev_mention2id.txt"
f_test_r2m = input_dir + "dev_rmpair.txt"

# trained KB file
f_relationvec = input_dir + 'relation2vec.bern'
f_entityvec = input_dir + 'entity2vec.bern'

# M2R+KB testing file
f_m2rkb = input_dir + 'dev.txt'

class BoW:
    def __init__(self, hyperPara):
        # 构建一个词袋(训练和测试时的词袋要一样啊！！！！！)
        self.vectorizer = CountVectorizer(dtype=np.float32)
        bow_mention_lst = set()
        self.mention_dct = dict()
        id = 0
        with open(f_train_m2id, "r") as f:
            for line in f:
                if id == 0:
                    id += 1
                    continue
                pair = line.split('\t')
                mention = pair[0]
                m_id = int(pair[1].strip('\n'))
                self.mention_dct[m_id] = mention
                bow_mention_lst.add(mention)
        if hyperPara.testFlag:
            id = 0
            self.mention_dct.clear()
            with open(f_test_m2id, 'r') as f:
                for line in f:
                    if id == 0:
                        id += 1
                        continue
                    pair = line.split('\t')
                    mention = pair[0]
                    m_id = int(pair[1].strip('\n'))
                    self.mention_dct[m_id] = mention
        X = self.vectorizer.fit_transform(bow_mention_lst)
        self.bow_size = X.shape[1]
    
    def get_embedding(self, id):
        # with sess.as_default():
        #     print(type(sess.run(id)))
        text = self.mention_dct[id]
        return self.vectorizer.transform([text]).toarray()
    
    def get_embedding_batch(self, ids):
        index = 0
        ret_np = None
        for id in range(ids.shape[0]):
            a = self.get_embedding(id)
            if index == 0:
                index += 1
                ret_np = a
            else:
                ret_np = np.vstack((ret_np, a))
        # print("******************************************")
        # print(np.transpose(ret_np))
        return np.transpose(ret_np)
        # return ret_np
        # print(ret_np)
        # raise IndexError()
        # return np.zeros((self.batch_size, self.bow_size))
class DataSet:
    def __init__(self, f_m2id, f_r2m):        
        self.mlst = []
        self.rlst = []
        self.r_given_m = dict()
        self.batch_id = 0
        self.batch_size = 100
        with open(relation2id, 'r') as f:
            self.relation_num = int(f.readline().strip('\n'))
        self.extract_data(f_m2id, f_r2m)
    def extract_data(self, f_m2id, f_r2m):
        with open(f_m2id, 'r') as f:
            self.mention_num = int(f.readline().strip('\n'))
        with open(f_r2m, 'r') as f:
            for line in f:
                pair = line.split("\t")
                r = int(pair[0])
                m = int(pair[1])
                self.mlst.append(m)
                self.rlst.append(r)
                if m not in self.r_given_m:
                    self.r_given_m[m] = set()
                self.r_given_m[m].add(r)
        self.data_size = len(self.mlst)
        self.nbatches = self.data_size // self.batch_size if self.data_size % self.batch_size == 0 else self.data_size // self.batch_size + 1
    def get_batch(self):
        batch_start = self.batch_id * self.batch_size
        temp = batch_start + self.batch_size
        if temp < self.data_size:
            batch_end = temp
        else:
            batch_end = self.data_size
            self.batch_id = 0
        m_i_batch = self.mlst[batch_start:batch_end]
        r_i_batch = self.rlst[batch_start:batch_end]
        return m_i_batch, r_i_batch
    
    

class TestSet(DataSet):
    def __init__(self):
        super().__init__(f_test_m2id, f_test_r2m)
        self.test_id = 0
        self.batch_size = self.relation_num
        self.rank_sum = 0
        self.hits_cnt = 0
    
    def get_relation_batch(self):
        ret_mention = [self.mlst[self.test_id] for i in range(self.relation_num)]
        ret_relation = [i for i in range(self.relation_num)]
        self.test_id += 1
        # if self.test_id == self.data_size:
        #     self.test_id = 0
        return ret_mention, ret_relation
    def evaluate(self, n, scores):
        score_dct = dict()
        self.n = n
        correct_id = self.rlst[self.test_id - 1]
        for i in range(scores.shape[0]):
            score_dct[i] = scores[i]
        sort_tup_lst = sorted(score_dct.items(), key=lambda item: item[1], reverse=True)
        rank = 0
        for i in range(len(sort_tup_lst)):
            if sort_tup_lst[i][0] == correct_id:
                rank = i + 1
                break
        self.rank_sum += rank
        if rank <= n:
            self.hits_cnt += 1
    
    def show_result(self):
        print('evaluating result, mean rank: {}, hits{}: {}'.format(
            self.rank_sum / self.data_size, self.n, self.hits_cnt / self.data_size))

class TrainSet(DataSet):
    def __init__(self):
        super().__init__(f_train_m2id, f_train_r2m)
    
    def get_batch(self):
        m_i_batch, r_i_batch = super().get_batch()
        m_j_batch = []
        r_n_batch = []
        mention_num = self.mention_num
        relation_num = self.relation_num
        for m_i in m_i_batch:
            m_j = random.randint(0, mention_num - 1)
            while m_j == m_i:
                m_j = random.randint(0, mention_num - 1)
            r_n = random.randint(0, relation_num - 1)
            while r_n in self.r_given_m[m_j] or r_n in self.r_given_m[m_i]:
                r_n = random.randint(0, relation_num - 1)
            m_j_batch.append(m_j)
            r_n_batch.append(r_n)
        return m_i_batch, r_i_batch, m_j_batch, r_n_batch

class KBModel:
    def __init__(self):
        self.read_vec(False)
        self.read_vec(True)
        self.hits_n = 10
        

    def norm2(self, h_vec, rel_vec, t_vec):
        return LA.norm(h_vec + rel_vec - t_vec)
    
    def evaluate_batch(self, h, t, ret_size):
        score_lst = []
        h_vec = self.e_vec[h]
        t_vec = self.e_vec[t]
        for i in range(self.r_num):
            rel_vec = self.r_vec[i]
            score_lst.append((i, self.norm2(h_vec, rel_vec, t_vec)))
        # sorted ascending
        sort_lst = sorted(score_lst, key=lambda item: item[1])
        ret_val = [0] * ret_size
        for i in range(len(sort_lst)):
            if i >= self.hits_n:
                break
            if sort_lst[i][0] < ret_size:
                ret_val[sort_lst[i][0]] = 1
        ret_val[0] = 0
        return ret_val
    # 判断答案是否在Hits10中
    def evaluate(self, h, r, t):
        score_lst = []
        h_vec = self.e_vec[h]
        t_vec = self.e_vec[t]
        for i in range(self.r_num):
            rel_vec = self.r_vec[i]
            score_lst.append((i, self.norm2(h_vec, rel_vec, t_vec)))
        # sorted ascending
        sort_lst = sorted(score_lst, key=lambda item: item[1])
        id = 0
        try:
            while id < self.hits_n:
                if r == sort_lst[id][0]:
                    return 1
                id += 1
        except:
            print("something goes wrong: {}".format(id))
            print("length of sort_lst: {}".format(len(sort_lst)))
            print("length of score_lst: {}".format(len(score_lst)))
        return 0

    def read_vec(self, entity=False):
        if entity:
            ff = f_entityvec
        else:
            ff = f_relationvec
        line_id = 0
        with open(ff, 'r') as f:
            mat = None
            for line in f:
                lst = re.split(r'\s+', line.strip('\n').strip())
                a = np.array([float(i) for i in lst])
                if line_id == 0:
                    mat = a
                else:
                    mat = np.vstack((mat, a))
                line_id += 1
            if entity:
                self.e_vec = mat
                self.e_num = line_id
                print("len of e_num: {}".format(self.e_num))
            else:
                self.r_vec = mat
                self.r_num = line_id
                print("len of r_num: {}".format(self.r_num))
        
class M2RKB:
    def __init__(self):
        self.test_id = 0
        self.m_given_h_t = dict()
        self.r_given_h_t = dict()
        with open(f_m2rkb, 'r') as f:
            for line in f:
                multi = [int(i) for i in line.strip('\n').split('\t')]
                rel = multi[0]
                if rel == 0:
                    # 不考虑NA
                    continue
                h = multi[1]
                t = multi[2]
                if (h,t) not in self.r_given_h_t:
                    self.r_given_h_t[(h,t)] = set()
                self.r_given_h_t[(h,t)].add(rel)
                if len(multi) < 4:
                    # 即没有mention
                    continue
                if (h,t) not in self.m_given_h_t:
                    self.m_given_h_t[(h,t)] = set()
                for i in range(3, len(multi)):
                    self.m_given_h_t[(h,t)].add(multi[i])
        self.h_t_lst = list(self.m_given_h_t.keys())
        self.m_lst = list(self.m_given_h_t.values())
        self.data_size = len(self.h_t_lst)
        with open(relation2id, 'r') as f:
            self.relation_num = int(f.readline().strip('\n'))
        self.precision = [0] * self.relation_num
        self.recall = [0] * self.relation_num
        self.correct_r_num = 0
        self.hit_num = 0
    def get_rel_mention_batch(self):
        m_set = self.m_lst[self.test_id]
        self.test_id += 1
        ret_mention = [i for i in m_set]
        ret_relation = [i for i in range(self.relation_num)]
        return ret_mention, ret_relation

    def m2r_kb_ev(self, scores, kbmodel):
        # scores 是一个num_mention行，num_rel行的矩阵，求每一列的和，得到r
        mention_sum = np.sum(scores, axis=0)
        h, t = self.h_t_lst[self.test_id - 1]
        mention_sum = mention_sum + kbmodel.evaluate_batch(h, t, self.relation_num)
        # for i in range(1, self.relation_num):
        #     mention_sum[i] += kbmodel.evaluate(h, i, t)
        score_lst = []
        for i in range(self.relation_num):
            score_lst.append((i, mention_sum[i]))
        rank_lst = sorted(score_lst, key=lambda item: item[1], reverse=True)
        correct_r = self.r_given_h_t[(h,t)]
        self.correct_r_num += len(correct_r)
        for i in range(1, len(rank_lst) + 1):
            answer_lst_size = i
            hit_cnt = 0
            for a in range(answer_lst_size):
                if rank_lst[a][0] in correct_r:
                    hit_cnt += 1
                    self.hit_num += 1
            self.precision[answer_lst_size - 1] += hit_cnt / answer_lst_size
            self.recall[answer_lst_size - 1] += hit_cnt / len(correct_r)
    
    def display_result(self):
        self.precision = [i / self.data_size for i in self.precision]
        self.recall = [i / self.data_size for i in self.recall]
        print(self.precision)
        print(self.recall)
        print(self.correct_r_num)
        print(self.hit_num)


    
    # def get_mention_batch(self, rel):
    #     m_set = self.m_lst[self.test_id - 1]
    #     ret_mention = [i for i in m_set]
    #     ret_relation = [rel]
    #     return ret_mention, ret_relation
    
    # def m2r_kb_ev(self, rel, kbmodel, scores):
    #     sigma_S_m2r = np.sum(scores)
    #     h, t = self.h_t_lst[self.test_id - 1]
    #     S_kb = kbmodel.evaluate(h, rel, t)
    #     self.rel = rel
        

class HyperPara:
    def __init__(self):
        self.learning_rate = 0.001
        self.dim = 50
        self.epoch = 50
        self.margin = 1.0
        # if you want to test, set these two options True
        self.loadFromData = True
        self.testFlag = True
        # use Sm2r+kb to evaluate relation extraction
        self.composite = True

class M2R:
    def __init__(self, dataset, bow, hyperPara, sess):
        relation_num = dataset.relation_num
        dim = hyperPara.dim
        margin = hyperPara.margin
        bow_size = bow.bow_size
        self.m_i = tf.placeholder(tf.int32, [None], name='m_i_p')
        self.r_i = tf.placeholder(tf.int32, [None], name='r_i_p')
        self.m_j = tf.placeholder(tf.int32, [None], name='m_j_p')
        self.r_n = tf.placeholder(tf.int32, [None], name='r_n_p')
        def get_bow_embedding(m):
            return tf.py_func(bow.get_embedding_batch, [m], tf.float32)
        with tf.name_scope("embedding"):
            # mention_embedding和relation_embedding
            self.w_embeddings = tf.get_variable(name="w_embeddings", shape=[bow_size, dim], initializer=tf.initializers.random_normal(0, 1/dim))
            self.r_embeddings = tf.get_variable(name="r_embeddings", shape=[relation_num, dim], initializer=tf.initializers.random_normal(0, 1/dim))
            self.w_embeddings = tf.nn.l2_normalize(self.w_embeddings)
            self.r_embeddings = tf.nn.l2_normalize(self.r_embeddings, axis=1)
            m_i_e = get_bow_embedding(self.m_i)
            r_i_e = tf.nn.embedding_lookup(self.r_embeddings, self.r_i)
            m_j_e = get_bow_embedding(self.m_j)
            r_n_e = tf.nn.embedding_lookup(self.r_embeddings, self.r_n)
        pos = tf.diag_part(tf.matmul(tf.transpose(tf.matmul(tf.transpose(self.w_embeddings), m_i_e)), tf.transpose(r_i_e)))
        neg = tf.diag_part(tf.matmul(tf.transpose(tf.matmul(tf.transpose(self.w_embeddings), m_j_e)), tf.transpose(r_n_e)))
        self.m2rkb_result = tf.matmul(tf.transpose(tf.matmul(tf.transpose(self.w_embeddings), m_i_e)), tf.transpose(r_i_e))
        self.predict = pos
        with tf.name_scope("output"):
            self.loss = tf.reduce_sum(tf.maximum(margin - pos + neg, 0))

def main(_):
    hyperPara = HyperPara()
    bow = BoW(hyperPara)
    if hyperPara.testFlag:
        dataset = TestSet()
    else:
        dataset = TrainSet()
    
    
    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer(uniform=False)
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                trainModel = M2R(dataset, bow, hyperPara, sess)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.GradientDescentOptimizer(hyperPara.learning_rate)
            grads_and_vars = optimizer.compute_gradients(trainModel.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
            saver = tf.train.Saver()
            sess.run(tf.initialize_all_variables())
            if hyperPara.loadFromData:
                saver.restore(sess, './model.vec')
            
            def train_step(m_i_batch, r_i_batch, m_j_batch, r_n_batch):
                feed_dict = {
                    trainModel.m_i: m_i_batch,
                    trainModel.m_j: m_j_batch,
                    trainModel.r_i: r_i_batch,
                    trainModel.r_n: r_n_batch
                }
                _, step, loss = sess.run([train_op, global_step, trainModel.loss], feed_dict)
                return loss

            def test_step(m_i_batch, r_i_batch):
                feed_dict = {
                    trainModel.m_i: m_i_batch,
                    trainModel.r_i: r_i_batch
                }
                predict = sess.run([trainModel.predict], feed_dict)
                return predict

            def sigma_m2r(m_i_batch, r_i_batch):
                feed_dict = {
                    trainModel.m_i: m_i_batch,
                    trainModel.r_i: r_i_batch
                }
                m2rkb_result = sess.run([trainModel.m2rkb_result], feed_dict)
                return m2rkb_result




            
            pmi = np.zeros(dataset.batch_size, dtype=np.int32)
            pri = np.zeros(dataset.batch_size, dtype=np.int32)
            pmj = np.zeros(dataset.batch_size, dtype=np.int32)
            prn = np.zeros(dataset.batch_size, dtype=np.int32)
            if not hyperPara.testFlag:
                for epoch in range(hyperPara.epoch):
                    res = 0.0
                    for batch in range(dataset.nbatches):
                        pmi, pri, pmj, prn = dataset.get_batch()
                        res += train_step(pmi, pri, pmj, prn)
                        current_step = tf.train.global_step(sess, global_step)
                    print(epoch)
                    print(res)
                saver.save(sess, "./model.vec")
            else:
                # data_size = dataset.data_size
                # for pair in range(data_size):
                #     # 用所有的relation去替换原来pair中的relation
                #     pmi, pri = dataset.get_relation_batch()
                #     scores = test_step(pmi, pri)
                #     dataset.evaluate(1, scores[0])
                # dataset.show_result()
                if hyperPara.composite:
                    m2rkb = M2RKB()
                    kbmodel = KBModel()
                    i = 0
                    for _ in range(m2rkb.data_size):
                        pmi, pri = m2rkb.get_rel_mention_batch()
                        print('***************************')
                        print(i)
                        print(pmi)
                        i += 1
                        scores = sigma_m2r(pmi, pri)
                        m2rkb.m2r_kb_ev(scores[0], kbmodel)
                    m2rkb.display_result()


if __name__ == "__main__":
	tf.app.run()    