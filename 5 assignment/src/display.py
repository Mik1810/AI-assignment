class Displayable(object):
	"""Class that uses 'display'.
	The amount of detail is controlled by max_display_level
	"""
	max_display_level = 4 # can be overridden in subclasses

	def display(self,level, *args, **nargs):
		"""print the arguments if level is less than or equal to the
		current max_display_level.
		level is an integer.
		the other arguments are whatever arguments print can take.
		"""
		if level <= self.max_display_level:
			with open("../console.txt", "a") as file:
				for arg in args:
					file.write(str(arg) + "\n")
			file.close()
			print(*args, **nargs) ##if error you are using Python2 not Python3


def visualize(func):
	"""A decorator for algorithms that do interactive visualization. Ignored here.
	"""
	return func



