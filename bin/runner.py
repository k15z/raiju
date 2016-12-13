from random import randint
from subprocess import call

def execute():
	size = str(randint(20, 50)) + " " + str(randint(20, 50))
	call(["./halite", "-d", size, "python ZBot2.py", "python ZBot3.py"])
	call(["./halite", "-d", size, "python ZBot3.py", "python ZBot3.py"])

for i in range(5):
	execute()
