import cProfile
from random import random
def filers():
	pass

def is_round(ellipse):
	center, (axis1,axis2), angle = ellipse
	ratio = .5
	if axis1 and axis2 and abs(1-axis2/axis1) < ratio:
		return True
	else:
		return False

def test():
	ellipses = [((20,12),(random(),random()),23.0) for x in xrange(40)]
	ratio = random()
	is_round_l = lambda (center, (axis1,axis2),angle): abs(1-axis2/axis1) < ratio
	# ellipses = filter(is_round_l,ellipses)
	# ellipses = [e for e in ellipses if is_round_l(e)]
	new = []
	for i in xrange(len(ellipses)):
		center, (axis1,axis2), angle = ellipses[i]
		if axis1 and axis2 and abs(1-axis2/axis1) < ratio:	
			new.append(ellipses[i])


if __name__ == '__main__':
    from timeit import Timer
    t = Timer("test()", "from __main__ import test")
    print t.timeit()