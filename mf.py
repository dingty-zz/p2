#!/usr/local/bin/python3
import sys
import os
import random
import time
import numpy as np
from pyspark import SparkContext
import pyspark
import collections as cl
x_block_dim = 0
y_block_dim = 0

N=16
# K=2
step_size = 0.001

def chunks(l, n):
	result = []
	step = len(l)/n
	for i in xrange(0, len(l), step):
		print i
		result.append(l[i:i+step])
	return result
def flatmap_add_index(line):
	row_index = line[1]
	row = line[0]
	for col_index in range(len(row)):
		yield col_index, (row_index, row[col_index])
def map_make_col(line):
	col_index = line[0]
	col = line[1]
	res = []
	for item in sorted(col):
		res.append(item[1])
	return col_index, np.array(res)
def transpose(matrix):
	matrix.ZipWithIndex().flatMap(flatmap_add_index).groupByKey() \
		.map(map_make_col)


def map_line(tuple):
	return tuple[1]
	# return np.array([tuple[0][0], tuple[0][1], tuple[1]])

# randomly initialize the factor matrices
def initialize_factor_matrices(N, K, x_block_dim, y_block_dim):
    random.seed(1) # always use the same seed to get deterministic results
    W = []
    for i in range(0, N):
        line = ""
        for j in range(0, x_block_dim):
            row = ""
            for k in range(0, K):
                row += str(random.random()) + ","
            line += row +";"
        W.append(parse_factor_matrix_line(line))

    H = []
    for i in range(0, N):
        line = ""
        for j in range(0, y_block_dim):
            row = ""
            for k in range(0, K):
                row += str(random.random()) + ","
            line += row +";"
        H.append(parse_factor_matrix_line(line))
    # print W
    # return W, H
    # return map(lambda x : parse_factor_matrix_line(x), W), map(lambda x : parse_factor_matrix_line(H))

def compute_strata():
	one_row = range(1, N)
	one_row.insert(0, N)
	pick_strata = []
	for i in range(0, N):
		pick_strata+=one_row
		one_row=[(lambda x : (x-1)%N)(k) for k in one_row]

	pick_strata = [(0 if k == N else k) for k in pick_strata]
	print pick_strata
	return pick_strata

# blockify the data matrix
def blockify_data(csv_file, N):
    max_x_id = 0
    max_y_id = 0
    # fobj = sc.textFile(csv_file).collect()
    #with hdfs.open(csv_file) as fobj:
    with open(csv_file) as fobj:
        for line in fobj:
            tokens = line.split(",")
            max_x_id = max(max_x_id, int(tokens[0]))
            max_y_id = max(max_y_id, int(tokens[1]))

    # assume the id starts from 0
    x_block_dim = int((max_x_id + N) / N)
    y_block_dim = int((max_y_id + N) / N)


    # create temporary data files
    tmp_files = []
    for i in range(0, N):
        files = []
        for j in range(0, N):
            # fobj = open(tmp_dir + "/data-" + str(i) + "-" + str(j) + ".csv", 'w+')
            files.append([])
        tmp_files.append(files)

    with open(csv_file) as fobj:
        for line in fobj:
            tokens = line.split(",")
            x_id = int(tokens[0])
            y_id = int(tokens[1])
            x_block_id = int(x_id / x_block_dim)
            y_block_id = int(y_id / y_block_dim)
                #print (x_block_id, y_block_id, x_id, y_id)
            tmp_files[x_block_id][y_block_id].append(line.strip())

    # block_fobj = open(data_block_file, 'w+')
    # for i in range(0, N):
    #     for j in range(0, N):
    #         out_line = ""
    #         with open(tmp_dir + "/data-" + str(i) + "-" + str(j) + ".csv", 'r') as fobj:
    #             for line in fobj:
    #                 tokens = line.split(",")
    #                 out_line += tokens[0] + "," + tokens[1] + "," + str(float(tokens[2])) + ";"
    #         block_fobj.write(out_line + "\n")
    # block_fobj.close()
    # print tmp_files
    result = reduce(lambda x, y:x+y, tmp_files)
    # print result
    return x_block_dim, y_block_dim, result

# one line is a partition of the factor matrix
def parse_factor_matrix_line(line):
    tokens = line.split(";")
    rows = []
    for token in tokens:
        row = []
        token = token.strip()
        if token == "":
            continue
        row_entries = token.split(",")
        for row_entry in row_entries:
            if row_entry == "":
                continue
            row.append(float(row_entry))
        rows.append(np.array(row))
    return rows

def sgd_on_one_block(x):
    # print "within a bokc"
    # print x

    offset_tuple = x[0][0][1]

    data_line_row = offset_tuple[0]
    data_line_col = offset_tuple[2]
    W_rows = x[0][1]
    W_rows_offset=offset_tuple[1]
    H_rows = x[1]
    H_rows_offset=offset_tuple[3]
    data_line = blocks_broadcast.value[data_line_row * N + data_line_col]
    # print "see what's going on"
    # print data_line
    # print
    # print W_rows
    # print W_rows_offset
    # print offset_tuple[2]
    # print H_rows
    # print H_rows_offset
    # print "donee"
    num_data_samples = 0

    for data_sample in data_line:
        if data_sample == "":
            continue
        tokens = data_sample.split(",")
        x_id = int(tokens[0])
        y_id = int(tokens[1])
        rating = float(tokens[2])

        diff = rating - np.dot(W_rows[x_id - W_rows_offset], H_rows[y_id - H_rows_offset])
        W_gradient = -2 * diff * H_rows[y_id - H_rows_offset]
        W_rows[x_id - W_rows_offset] -= step_size * W_gradient

        H_gradient = -2 * diff * W_rows[x_id - W_rows_offset]
        H_rows[y_id - H_rows_offset] -= step_size * H_gradient
        num_data_samples += 1

    # print W_rows
    # print H_rows
    # print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    return (W_rows, H_rows)

def factor_matrix_rows_to_string(rows):
    # print "oi"
    row_len = len(rows)
    # print row_len
    idx=0
    line = ""
    for row in rows:
        idx += 1
        for num in np.nditer(row):
            line += str(num) + ","
        if idx != row_len:
            line += "\n"
    # print line
    return line

# perform evaluation of the model one block at a time
def evaluate_block_by_block(N, x_block_dim, y_block_dim):
    block_fobj = open(data_block_file, 'r')
    W_fobj = open(W_filename, 'r')
    error_total = .0
    n_total = 0

    # iterate over rows
    for i in range(0, N):
        W_line = W_fobj.readline()
        W_rows = parse_factor_matrix_line(W_line)
        W_rows_offset = i * x_block_dim
        H_fobj = open(H_filename, 'r')

        # iterate over blocks on the same row
        for j in range(0, N):
            data_line = block_fobj.readline()
            H_line = H_fobj.readline()
            H_rows = parse_factor_matrix_line(H_line)
            H_rows_offset = j * y_block_dim

            err, n = evaluate_on_one_block(data_line, W_rows, W_rows_offset,
                                        H_rows, H_rows_offset)
            error_total += err
            n_total += n
    return error_total, n_total

def create_tuple(line):
	tokens = line.split(",")
	# parse the original data line, which is (row_id, column_id, value)
	return (int(tokens[0]), int(tokens[1])), float(tokens[2])
if __name__ == '__main__':
    conf = pyspark.SparkConf().setAppName("mf")
    sc = pyspark.SparkContext(conf = conf)
    # load RDD from hdfs

    # csv_file = "toy.csv"

    csv_file = sys.argv[1]
    K = int(sys.argv[2]) #rank
    w_location = sys.argv[3]
    h_location = sys.argv[4]

    num_iterations = 10
    eta_decay = 0.99
    x_block_dim, y_block_dim, blocks = blockify_data(csv_file, N)
    W, H = initialize_factor_matrices(N, K, x_block_dim, y_block_dim)

    # print blocks
    blocks_broadcast = sc.broadcast(blocks)
    # print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
    # print W
    # print H
    W = sc.parallelize(W, N)
    H = sc.parallelize(H, N)
    tuples = []
    for i in range(0, N):
        W_rows_offset = i * x_block_dim
        for j in range(0, N):
            H_rows_offset = j * y_block_dim
            tuples.append((i,W_rows_offset, j , H_rows_offset))


    # print tuples

    strata = compute_strata()
    # print strata
    # strata[0]=1
    datas = zip(strata, tuples)
    # print datas
    for iterator in range(0, num_iterations):
    	for strata_idx in xrange(0,N):
            temp = sc.parallelize(filter(lambda x : x[0]==strata_idx, datas),N)
    	    a_strata = temp.zip(W).zip(H)
    	    a_strata.cache()
    	    # print "strariaaaaaaaaa"
    	    # print W
    	    # print H
    	    # H_broadcast = sc.broadcast(H)
            result = a_strata.map(lambda x : sgd_on_one_block(x))
            # a_strata.unpersist()
            # W.unpersist()
            # H.unpersist()
            W = result.map(lambda x : x[0])
            H = result.map(lambda x : x[1])
            # print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"

    	    # updated = H.collect()
    	    # for x in updated:
    	    # 	# W[x[0]] = x[1]
    	    # 	H[x[1]] = x[2]
            H_collected = H.collect()
            # print len(H_collected)
            # result.unpersist()

            H_list = cl.deque(H_collected)
            H_list.rotate(1)
            # H.unpersist()
            H = sc.parallelize(list(H_list), N)
            W.cache()
            H.cache()
    	step_size *= eta_decay

	    	# print x[1]
	    	# print x[3]
		# print "###################################################################"
# print "final w is"
# print W.collect()
# print "final H is"
# print H.collect()
# print "final w is"

Wresult = W.coalesce(1).map(lambda x : factor_matrix_rows_to_string(x))
Hresult = H.coalesce(1).map(lambda x : factor_matrix_rows_to_string(x))
Wresult.saveAsTextFile(w_location)
Hresult.saveAsTextFile(h_location)


# Wresult = W.collect()
# W.unpersist()
# # print "final H is"
# Hresult = H.collect()
# H.unpersist()

# W_file = open(w_location, 'w+')
# H_file = open(h_location, 'w+')

# print Wresult
# for w in Wresult:
# 	line = factor_matrix_rows_to_string(w)
# 	W_file.write(w + '\n')
# for h in Hresult:
# 	line = factor_matrix_rows_to_string(h)
# 	H_file.write(h + '\n')





	# scores = sc.textFile("toy.csv").map(create_tuple)
	# sorted_key = scores.sortByKey(True,N**2)
	# raw_data = sorted_key.collect()
	# max_x_id = sorted_key.map(lambda x: x[0][0]).max()
	# max_y_id = sorted_key.map(lambda x: x[0][1]).max()
 # 	print "max id (%d, %d)" % (max_x_id, max_y_id)
 #    # assume the id starts from 0
	# x_block_dim = int((max_x_id + N) / N)
	# y_block_dim = int((max_y_id + N) / N)
	# print "x, y block dim(%d, %d)" % (x_block_dim, y_block_dim)
	# # raw_data = sorted_key.map(map_line).collect()
	# row_partitioned = sc.parallelize(chunks(raw_data, N), N)
	# row_partitioned2 = row_partitioned.map(lambda x : chunks(x, N))
	# rp3 = row_partitioned2.map(lambda x: list(map(list, zip(*x))))
	# rp4 = rp3.map(lambda x: reduce(lambda a, b : a+b,x))
	# rp5 = rp4.map(lambda x : chunks(x, N))
	# blocks_with_indexes = rp5.reduce(lambda a, b : a+b)

	# blocks_par = sc.parallelize(blocks_with_indexes, N)
	# blocks1 = blocks_par.map(lambda x : map(lambda y : map_line(y), x))
	# blocks = blocks1.collect()
