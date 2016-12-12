from random import randint
from subprocess import call

def execute():
	size = str(randint(20, 50)) + " " + str(randint(20, 50))
	call(["./bin/halite", "-d", size, "python bin/ZBot1.py", "python bin/ZBot2.py"])

for i in range(2):
	execute()
