from psycopg2 import sql
import numpy as np
from sklearn import preprocessing
from sklearn.externals import joblib
import torch
import torch.utils.data
from werkzeug.exceptions import InternalServerError
import torch.nn.functional as F
import time
import os

# from kartoffel import database

class Dataset():
	def __init__(self, db, name):
		self.name = name

		self.db = db
		# select columns
		query = sql.SQL("""SELECT column_name
	FROM information_schema.columns
	WHERE table_schema = {schema}
	AND NOT (column_name = 'mask')
	AND NOT (column_name = 'date')
	AND table_name   = {tbl}
	""").format(schema=sql.Literal(db.config['schema']), tbl=sql.Literal(name))
		try:
			rows = db.query(query)
		except:
			raise InternalServerError(description="Failed to select the column_name of an table.")

		self.columns_all = [r[0] for r in rows]
		self.columns = [r[0] for r in rows]
		# # 測試使用
		# self.columns = ['temp', 'hum', 'windspeed', 'casual', 'registered', 'cnt']
		# # 
		self.dataloader = None
		# self.hdd = None
		print('new dataset: {0}'.format(name))
		self.mean = 0
		
	def is_valid_columns(self, columns):
		return all([c in self.columns_all for c in columns])

	def set_columns(self, columns):
		if not columns:
			raise InternalServerError(
				description='Provided columns is an empty list. Expected a length >= 1 list')

		if not self.is_valid_columns(columns):
			raise InternalServerError(description="Some columns doesn't exist")

		self.columns = columns

	def distance(self,data):
		loss_fun = torch.nn.MSELoss(reduction='none')
		num = data.shape[0]
		dim = data.shape[1]
		distance_matrix = []
		for i in range(num):
			temp = torch.zeros(num,dim)
			d = data[i]
			d = d.repeat(num,1)
			d = loss_fun(data,d)
			d = torch.sum(d,dim=1)
			d = torch.sqrt(d)
			d[:i] = 0
			distance_matrix.append(d.view(1,-1))
		return torch.cat(distance_matrix,dim=0)

	def distance_mean(self,data, eps=1e-8):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		n = data.shape[0]
		mean = []
		distance_matrix = []
		data = data.to(device)
		for i in range(n):
			dist = data - data[i:i+1].repeat(n, 1)
			dist = dist.pow(2).sum(1)
			dist = (dist+eps).sqrt()
			dist[i] = 0
			distance_matrix.append(dist)
		distance_matrix = torch.cat(distance_matrix,0).view(n,n)
		distance_matrix = distance_matrix.cpu()
		mask = torch.ones(n,n)
		mask = torch.triu(mask,diagonal=1)
		mean = distance_matrix[mask>0].mean()
		distance_matrix[mask>0] = distance_matrix[mask>0]/mean
		return mean, distance_matrix[mask>0]

	def load(self, batch_size, input_window, output_window, normalizer=None, all_level=False):
		# load raw data
		rawdata, _ = self.select(get_date=True, date_range=None)
		rawdata = np.array(rawdata)
		dates = rawdata[:, 0]
		rawdata = rawdata[:, 1:].astype('float32')

		# normalize
		if normalizer is None:
			try:
				normalizer = preprocessing.MinMaxScaler((0, 1)).fit(rawdata)
			except:
				raise InternalServerError(description="An error occur during normalization.")
		# joblib.dump(normalizer, 'normalizer.pkl')
		# train_dataset = n * dimension_num
		train_dataset = torch.Tensor(normalizer.transform(rawdata))

		if all_level:
			########################
			## add the high level data to raw data, like: 1hour -> 2hour 
			########################
			X_test_day = self.loadLowLevel(interval='1 day')
			X_test_hour = self.loadLowLevel(interval='2 hour')
			X_test_day = np.array(X_test_day)[:, 1:]
			X_test_hour = np.array(X_test_hour)[:, 1:]
			X_test_day = torch.Tensor(normalizer.transform(X_test_day))
			X_test_hour = torch.Tensor(normalizer.transform(X_test_hour))

			train_dataset = torch.cat((train_dataset,X_test_day,X_test_hour),0)
			print("##################total train_dataset")		
		# prepare window
		# train_dataset shape n*dimension
		si = 0
		ei = train_dataset.shape[0] + 2 - (input_window+output_window)
		si -= 1
		ei -= 1
		align = []
		for i in range(input_window):
			si += 1
			ei += 1
			align.append(train_dataset[si:ei].clone())
		input_dataset = torch.cat(align, dim=1)

		align = []
		for i in range(output_window):
			align.append(train_dataset[si:ei])
			si += 1
			ei += 1
			
		output_dataset = torch.cat(align, dim=1)

		all_dataset = torch.cat([input_dataset, output_dataset], dim=1)
		split_idx = all_dataset.shape[1]
		# torch.save(all_dataset[:, :split_idx],self.name)

		PRE_COMPUTE_MEAN = './distance_map/'+self.name+'_pre_compute_mean.pth'
		PRE_COMPUTE_DISTANCE_MATRIX = './distance_map/'+self.name+'_pre_compute_distance_matrix.pth'

		if not os.path.exists(PRE_COMPUTE_MEAN):
			self.mean, distance_matrix = self.distance_mean(data=all_dataset[:, :split_idx//2])
			torch.save((self.mean), PRE_COMPUTE_MEAN)
			torch.save((distance_matrix), PRE_COMPUTE_DISTANCE_MATRIX)
			del distance_matrix
		else:
			self.mean = torch.load(PRE_COMPUTE_MEAN)
		
		dataloader = torch.utils.data.DataLoader(all_dataset,
			batch_size=batch_size,
			shuffle=True,
			num_workers=1,
			pin_memory=True)

		self.normalizer = normalizer
		self.dataloader = dataloader
		self.train_dataset = train_dataset
		self.dates = dates
		self.rawdata = rawdata

	def load2(self, batch_size, input_window, output_window):
		# load raw data
		rawdata, _ = self.select(get_date=True, date_range=None)
		rawdata = np.array(rawdata)
		rawdata = rawdata[:, 1:].astype('float32')
		# normalize
		try:
			normalizer = preprocessing.MinMaxScaler((0, 1)).fit(rawdata)
		except:
			raise InternalServerError(description="An error occur during normalization.")
		self.normalizer = normalizer

	def get_input(self):
		# load raw data
		rawdata, _ = self.select(get_date=True, date_range=None)
		rawdata = np.array(rawdata)
		rawdata = rawdata[:, 1:].astype('float32')
		# normalize
		try:
			normalizer = preprocessing.MinMaxScaler((0, 1)).fit(rawdata)
		except:
			raise InternalServerError(description="An error occur during normalization.")

		input_dataset = torch.Tensor(normalizer.transform(rawdata))
		del rawdata
		# del normalizer
		return normalizer, input_dataset

	def loadLowLevel(self,interval,date_range=None):
		data_test,_ = self.select_interval(
			interval = interval,
			date_range = date_range,
			columns = self.columns,
			get_mask=False
		)
		return data_test

	def load_day(self, normalizer, date_range=None):
		X_test_day = self.loadLowLevel('1 day', date_range)
		X_test_day = np.array(X_test_day)[:, 1:]
		X_test_day = torch.Tensor(normalizer.transform(X_test_day))
		return X_test_day
		

	def select(self, get_date=False, date_range=None, columns=None):
		"""Select data from table with date and all the selected columns
		Args:
			get_date (bool): select date field or not
			date_range (tuple(start, end)): select within a date range. if this is true, start_date and end_date
				sould also be set.
			columns ([str]): a list of selected columns. default is self.columns

		Returns:
			result from database query
		"""

		db = self.db
		if date_range:
			date_snip = sql.SQL('WHERE date >= {sd} AND date <= {ed}').format(
				sd=sql.Literal(date_range[0]),
				ed=sql.Literal(date_range[1])
			)
		else:
			date_snip = sql.SQL('')

		if columns is None:
			columns = self.columns

		if get_date:
			columns = ['date'] + columns

		columns_snip = sql.SQL(',').join(sql.Identifier(c) for c in columns)

		query = sql.SQL("""
WITH filtered_table AS (SELECT * FROM {schema}.{tbl} WHERE mask = FALSE)
SELECT {columns} FROM filtered_table {date_cond}
		""").format(
			schema=sql.Identifier(db.config['schema']),
			tbl=sql.Identifier(self.name),
			columns=columns_snip,
			date_cond=date_snip,
		)
		try:
			res = db.query(query)
		except:
			raise InternalServerError(description="Failed to select data.")
		return res, columns

	def select_interval(self, interval=None, date_range=None, columns=None, get_mask=True):
		db = self.db
		if interval is None:
			raise InternalServerError(description='interval is needed')

		if date_range:
			date_snip = sql.SQL('WHERE date >= {sd} AND date <= {ed}').format(
				sd=sql.Literal(date_range[0]),
				ed=sql.Literal(date_range[1])
			)
		else:
			date_snip = sql.SQL('')

		if columns is None:
			columns = self.columns

		columns_snip = sql.SQL(',').join(
			sql.SQL("AVG({})").format(sql.Identifier(c)) for c in columns
		)

		if get_mask:
			mask_snip = sql.SQL(", count(CASE WHEN mask THEN 1 END)")
		else:
			mask_snip = sql.SQL('')

		query = sql.SQL("""
		WITH const AS (SELECT extract(EPOCH from {interval}::interval) gap)
SELECT to_timestamp(floor((extract(EPOCH from date) / const.gap)) * const.gap)
AT TIME ZONE 'UTC' as interval_alias,
{grouped_columns} {mask}
FROM {schema}.{tbl}, const
{date_cond}
GROUP BY interval_alias
ORDER BY interval_alias
		""").format(
			schema=sql.Identifier(db.config['schema']),
			tbl=sql.Identifier(self.name),
			interval=sql.Literal(interval),
			grouped_columns=columns_snip,
			mask=mask_snip,
			date_cond=date_snip
		)
		try:
			res = db.query(query)
		except:
			raise InternalServerError(description="Failed to select data with interval.")
		return res, ['date'] + columns + (['mask'] if get_mask else [])

def getDefault():
	db = self.db
	datasets = db.listall()
	if datasets:
		return Dataset(datasets[0])
	raise InternalServerError(description="There's no dataset yet.")
