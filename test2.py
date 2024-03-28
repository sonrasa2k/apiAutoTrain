from rq import Queue,Worker
from redis import Redis

conn1 = Redis('localhost', 6379)
conn2 = Redis('remote.host.org', 9836)

q1 = Queue('foo', connection=conn1)
q2 = Queue('bar', connection=conn2)
worker1 = Worker([q1], connection=conn1, name='foo')
worker1.work()
# worker2 = Worker([q2], connection=conn2, name='bar')