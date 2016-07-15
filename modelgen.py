# -*- coding: utf-8 -*-

'''
modelgen.py by Wanja Hentze

To the extent possible under law, the person who associated CC0 with
modelgen.py has waived all copyright and related or neighboring rights
to modelgen.py.

You should have received a copy of the CC0 legalcode along with this
work.  If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
'''

from math import *
from random import seed, random, randint, choice, uniform, gauss
from numpy import mean

def isbad(f):
	return (isnan(f) or isinf(f))

class expression:
	def evaluate(x):
		raise("wat")

## binary expressions
class bin_exp(expression):
	def __init__(self, complexity, depth, dim, symbols, varnames):
		self.varnames = varnames
		self.dim = dim
		leftcomp   = randint(0, complexity-2)
		rightcomp  = complexity-2 - leftcomp 
		self.left  = gen_exp(leftcomp,  depth-1, dim, symbols, varnames)
		self.right = gen_exp(rightcomp, depth-1, dim, symbols, varnames)
		self.op = choice(symbols.bin_ops)()

	def evaluate(self,values):
		return self.op.apply(self.left.evaluate(values), self.right.evaluate(values))
	
	def pr(self):
		return "(" + self.left.pr() + " " + self.op.pr() + " " + self.right.pr() + ")"
	
	def isconst(self):
		return self.left.isconst() and self.right.isconst()
	
	def constreplace(self):
		if(self.left.isconst() and not isinstance(self.left.op, namedconstant)):
			val = self.left.evaluate({x:0.0 for x in self.varnames})
			self.left = term_exp(symboltable, self.varnames)
			self.left.op = constant(val)
		else:
			self.left.constreplace()
		
		if(self.right.isconst() and not isinstance(self.right.op, namedconstant)):
			val = self.right.evaluate({x:0.0 for x in self.varnames})
			self.right = term_exp(symboltable, self.varnames)
			self.right.op = constant(val)
		else:
			self.right.constreplace()

class plus_op():
	def apply(self, x, y):
		return x + y

	def pr(self):
		return "+"

class mult_op():
	def apply(self, x, y):
		return x * y
	
	def pr(self):
		return "*"

class minus_op():
	def apply(self, x, y):
		return x - y
	
	def pr(self):
		return "-"

class div_op():
	def apply(self, x, y):
		if (y == 0):
			return x * float('NaN')
		return x/y
	
	def pr(self):
		return "/"


## unary expression
class un_exp(expression):
	def __init__(self, complexity, depth, dim, symbols, varnames):
		self.varnames=varnames
		self.dim=dim
		self.arg = gen_exp(complexity - 1, depth - 1, dim, symbols, varnames)
		self.op = choice(symbols.un_ops)()
	
	def evaluate(self, values):
		return self.op.apply(self.arg.evaluate(values))
	
	def pr(self):
		return self.op.pr1() + self.arg.pr() + self.op.pr2()

	def isconst(self):
		return self.arg.isconst()
	
	def constreplace(self):
		if(self.arg.isconst() and not isinstance(self.arg.op, namedconstant)):
			val = self.arg.evaluate({x:0.0 for x in varnames})
			self.arg = term_exp(symboltable)
			self.arg.op = constant(val)
		else:
			self.arg.constreplace()

class sin_op():
	def apply(self, x):
		if isbad(x):
			return float('NaN')
		return sin(x)
	
	def pr1(self):
		return "sin("
	def pr2(self):
		return ")"

class cos_op():
	def apply(self, x):
		if isbad(x):
			return float('NaN')
		return cos(x)
	
	def pr1(self):
		return "cos("
	def pr2(self):
		return ")"

class sqrt_op():
	def apply(self, x):
		if not (x > 0):
			return float('NaN')
		return sqrt(x)
	
	def pr1(self):
		return "√"
	def pr2(self):
		return ""

class exp_op():
	def apply(self, x):
		if isbad(x) or (x > 709.5):
			return(float('NaN'))
		return exp(x)
	
	def pr1(self):
		return "exp("
	def pr2(self):
		return ")"

class square_op():
	def apply(self, x):
		return x*x
	
	def pr1(self):
		return ""
	def pr2(self):
		return "²"

class log_op():
	def apply(self, x):
		if (x <= 0):
			return float('NaN')
		return log(x)
	
	def pr1(self):
		return "ln("
	def pr2(self):
		return ")"

class acos_op():
	def apply(self, x):
		return acos(x)
	
	def pr1(self):
		return "acos("
	def pr2(self):
		return ")"

class abs_op():
	def apply(self, x):
		return abs(x)
	
	def pr1(self):
		return "|"
	def pr2(self):
		return "|"

class cosh_op():
	def apply(self, x):
		if isbad(x) or (abs(x) > 710.4):
			return float('NaN')
		return cosh(x)
	
	def pr1(self):
		return "cosh("
	def pr2(self):
		return ")"

class neg_op():
	def apply(self, x):
		if(isbad(x)):
			return float('NaN')
		return -x
	
	def pr1(self):
		return "-"
	def pr2(self):
		return ""

## terminal expressions
class term_exp(expression):
	def __init__(self, symbols, varnames):
		self.dim=len(varnames)
		self.op = choice(symbols.term_ops)(varnames)

	def evaluate(self, values):
		return self.op.apply(values)
	
	def pr(self):
		return self.op.pr()

	def isconst(self):
		return self.op.isconst()
	
	def constreplace(self):
		return

class term_op():
	pass

class variable(term_op):
	def __init__(self, varnames):
		self.symbol = choice(varnames)
	
	def apply(self, values):
		return values[self.symbol]
	
	def pr(self):
		return self.symbol
	
	def isconst(self):
		return False

class constant(term_op):
	def __init__(self, value):
		self.value = value
	
	def apply(self, x):
		return self.value
	
	def pr(self):
		return " " + str(self.value)
	
	def isconst(self):
		return True

def randomconstant(varnames):
	return constant(uniform(-5.0, 5.0))

namedconstants = [
	("e", e),
	("π", pi),
	("φ", (1+sqrt(5))/2),
	("2", 2)
]

class namedconstant(term_op):
	def __init__(self, varnames):
		self.name, self.value = choice(namedconstants)
	
	def apply(self, values):
		return self.value
	
	def pr(self):
		return self.name
	
	def isconst(self):
		return True

class mult_variable(term_op):
	def __init__(self, varnames):
		self.varname = choice(varnames)
		self.mult = uniform(-5.0, 5.0)
	
	def apply(self, values):
		return self.mult * values[self.varname]
	
	def pr(self):
		return "(" + str(self.mult) + '*' + self.varname + ")"
	
	def isconst(self):
		return False

## default list of symbols
class symboltable:
	bin_ops = [plus_op, mult_op, minus_op, div_op]
	un_ops  = [sin_op, cos_op, sqrt_op, exp_op, abs_op, square_op, log_op, cosh_op]
	term_ops= [randomconstant, namedconstant, variable, variable, mult_variable]

## generate an expression with maximum complexity and depth, using symbols from symbol table
def gen_exp(complexity, depth, dim, symbols, varnames):
	if (complexity == 0 or depth == 0):
		return(term_exp(symbols, varnames))
	
	if (complexity == 1 or randint(0, 2) == 0):
		return(un_exp(complexity, depth, dim, symbols, varnames))
	else:
		return(bin_exp(complexity, depth, dim, symbols, varnames))

## evaluate exp num_samples times in an interval from xmin to xmax
def sample_exp(exp, num_samples=100, xmin=-4.0, xmax=4.0, regular=True):
	samples = list(dict())
	if(exp.dim == 1):
		if (regular):
			samples = [{exp.varnames[0]:xmin + ((xmax-xmin)/num_samples * i)}
				for i in range (num_samples)]
		else:
			samples = sorted([{exp.varnames[0]:uniform(xmin, xmax)}
				for i in range(num_samples)])
	elif(exp.dim == 2):
		if (regular):
			num1 = int(sqrt(num_samples))
			num2 = int(num_samples/num1)
			samples = [
				{exp.varnames[0]:xmin+((xmax-xmin)/num1 * x),
				 exp.varnames[1]:xmin+((xmax-xmin)/num2 * y)}
				for x in range(num1) for y in range(num2)]
		else:
			samples = [{exp.varnames[0]:uniform(xmin,xmax),
						exp.varnames[1]:uniform(xmin,xmax)}
						for i in range(num_samples)]
	
	return [(x, exp.evaluate(x)) for x in samples]

def addnoise(data, noise_type='additive', noise_amount=0.1):
	for tup in data:
		if noise_type == 'additive':
			tup[1] += uniform(-noise_amount, noise_amount)
		elif noise_type == 'multiplicative':
			tup[1] *= 1 + uniform(-noise_amount, noise_amount)
		elif noise_type == 'gaussian':
			tup[1] += gauss(0, noise_amount)
		
## repeatedly generate expressions until one yeilds no NaN, inf, constant or huge values
def modelgen(complexity, depth, dim = 1, symbols = symboltable(),
		varnames=['x'], constreplace = True):
	## only 1- and 2-dimensional functions supported for now
	assert(dim in [1,2])
	assert(len(varnames)==dim)
	
	while(1):
		exp = gen_exp(complexity, depth, dim, symbols, varnames)
		## try exp on some inputs to see if it's well-behaved
		## we only want non-constant, non-huge and well-defined models
		if(dim==1):
			sams = sample_exp(exp, 200)
			isconstant = (sams[0][1] == sams[10][1])
		else:
			sams = sample_exp(exp, 1600)
			isconstant = (sams[0][1] == sams[10][1] or
				sams[0][1] == sams[20][1])
		
		hugesample = (max([abs(s[1]) for s in sams]) > 100000)
		badsample = any([isbad(s[1]) for s in sams])
		if not (isconstant or hugesample or badsample):
			break
	if constreplace:
		exp.constreplace()
	return exp

def writearraytoffx(data, basename):
	fin  = open(basename + '_in.csv', 'w')
	fout = open(basename + '_out.csv', 'w')
	for tup in data[:-1]:
		fin.write(str(tup[0])+'\n')
		fout.write(str(tup[1])+' ')
	## no trailing space after last entry
	fin.write(str(data[-1][0]))
	fout.write(str(data[-1][1]))
	fin.close
	fout.close

## representation structure for sample data
class data:
	def __init__(self, samples, expression, name):
		self.samples = samples
		self.expression = expression
		self.name = name
	
	## simple, single csv file output
	def writetocsv(self, filename, writenames=False):
		f = open(filename, 'w')
		varnames = sorted(self.samples[0][0].keys())
		if(writenames):
			f.write(', '.join(varnames) + ', f\n')
		for tup in self.samples:
			ins = tup[0]
			out = tup[1]
			f.write(', '.join([str(ins[x]) for x in varnames]))
			f.write(', {0:12f}\n'.format(out))
		f.close
	
	## write to 4 separate files as required by ffx,
	## separating training/testing data and input/output
	def writeforffx(self, basename, testportion=0.25):
		cutoff = int(testportion*len(self.samples))
		traindata = self.samples[:cutoff]
		testdata  = self.samples[cutoff:]
		writearraytoffx(traindata, basename+'_train')
		writearraytoffx(testdata,  basename+'_test')

## print a list of 2-tuples csv-style
def printcsv(res):
	s = ''
	for tup in res:
		s += '{0:10f}, {1:20f}\n'.format(tup[0], tup[1])
	return s
