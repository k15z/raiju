from random import randint
from subprocess import call

def execute():
	size = str(randint(20, 50)) + " " + str(randint(20, 50))
	call(["./halite", "-d", size, "python ZBot1.py", "python Raiju.py"])
	call(["./halite", "-d", size, "python ZBot2.py", "python Raiju.py"])
	call(["./halite", "-d", size, "python ZBot3.py", "python Raiju.py"])

execute()
