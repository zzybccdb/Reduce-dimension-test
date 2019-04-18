import os
import io
import psycopg2
from psycopg2 import sql
import pandas as pd
from flask import current_app, g
from werkzeug.exceptions import InternalServerError

SCHEMA = os.getenv('SCHEMA')

class WISEPaaSPostgreSQL():
	def __init__(self):
		pass

	def setUp(self):
		config = {
			'host': '140.110.5.72',
			'user': '44352c6e-5d46-4e2b-a27c-565ecf11e0ec',
			'password': 'oileco7orrp49cadqg371hom3f',
			'dbname': 'baa87699-e895-43bd-80c2-81e3810d75fc',
			'sslmode': 'prefer',
			'schema': 'economic',
		}
		conn_string = "host={host} user={user} dbname={dbname} password={password} sslmode={sslmode}"
		try:
			self.conn = psycopg2.connect(conn_string.format(**config))
		except:
			raise InternalServerError(description="Failed to connect to the Database.")
		self.config = config

	def tearDown(self):
		self.conn.close()

	def query(self, querystr, *data):
		cursor = self.conn.cursor()
		ret = None
		try:
			if data:
				cursor.execute(querystr, data)
			else:
				cursor.execute(querystr)
			ret = cursor.fetchall()
		except psycopg2.Error as e:
			current_app.logger.error(str(e))
			self.conn.rollback()
			raise Exception('db.query() error')
		finally:
			cursor.close()

		return ret

	def execute(self, querystr, *data):
		cursor = self.conn.cursor()
		try:
			if data:
				cursor.execute(querystr, data)
			else:
				cursor.execute(querystr)
			self.conn.commit()
		except psycopg2.Error as e:
			print(e)
			self.conn.rollback()
			raise Exception('db.execute() error')
		finally:
			cursor.close()

	def listall(self):
		qs = "SELECT tablename FROM pg_catalog.pg_tables where schemaname=%s;"
		try:
			rows = self.query(qs, SCHEMA)
		except:
			raise InternalServerError(description="Failed to select table names from schema.")
		return [r[0] for r in rows]

	def create_dataset_from_csv(self, df, name='__testtbl__'):
		current_app.logger.info('Dropping N/A records')
		current_app.logger.info('Before cleaning shape={0}'.format(df.shape))
		df = df.dropna(axis=0, how='all')
		df = df.fillna(method='ffill')
		current_app.logger.info('After cleaning shape={0}'.format(df.shape))

		columns = list(df.columns)[1:]
		current_app.logger.info('columns: {0}'.format(columns))


		current_app.logger.info('create table: {0}'.format(name))
		query = sql.SQL("""CREATE TABLE {schema}.{tbl}
	(
		date TIMESTAMP WITHOUT TIME ZONE NOT NULL,
		{columns},
		"mask" BOOLEAN DEFAULT FALSE,
		PRIMARY KEY (date)
	)
	WITH (
		OIDS = FALSE
	);
	ALTER TABLE {schema}.{tbl}
		OWNER to {user};
	""").format(
			schema=sql.Identifier(self.config['schema']),
			tbl=sql.Identifier(name),
			columns=sql.SQL(',').join(
				sql.SQL("{0} double precision").format(sql.Identifier(c)) for c in columns
			),
			user=sql.Identifier(self.config['user']),
		)
		try:
			self.execute(query)
		except:
			raise InternalServerError(description="Fail to create create new table.")

		current_app.logger.info('uploading csv')
		cursor = self.conn.cursor()
		query = sql.SQL(
	"""COPY {schema}.{tbl} ({columns})
	FROM STDIN WITH CSV HEADER DELIMITER AS ','""").format(
					schema=sql.Identifier(self.config['schema']),
					tbl=sql.Identifier(name),
					columns=sql.SQL(',').join(
						sql.Identifier(c) for c in ['date'] + columns
					),
				)
		try:
			csv = io.StringIO()
			df.to_csv(csv, index=False)
			# this is the tricky part
			# you have to seek back to the start before being read
			# otherwise it will continue to read from the end
			# which reads nothing
			csv.seek(0)
			cursor.copy_expert(sql=query, file=csv)
			self.conn.commit()
		except:
			self.conn.rollback()
			raise InternalServerError(description="Failed to write data into table")
		finally:
			cursor.close()

		current_app.logger.info('success')

	def delete_dataset(self, name='__testtbl__'):
		current_app.logger.info('deleting table: {0}'.format(name))
		query = sql.SQL('DROP TABLE {schema}.{tbl}').format(
			schema=sql.Identifier(self.config['schema']),
			tbl=sql.Identifier(name)
		)
		try:
			self.execute(query)
		except:
			raise InternalServerError(description="Failed to delete dataset.")

		current_app.logger.info('success')


DEFAULT_DB = WISEPaaSPostgreSQL

def get_db():
	if 'db' not in g:
		g.db = DEFAULT_DB()
		g.db.setUp()
	return g.db

def close_db(what):
	db = g.pop('db', None)
	if db is not None:
		db.tearDown()

def init_app(app):
	app.teardown_appcontext(close_db)
