from scipy import misc
import numpy
import argparse
from itertools import combinations_with_replacement, permutations, product
#from colormath.color_diff import delta_e_cie2000
from random import randint, choice, shuffle, sample
import sys
import copy
import os
import operator
from collections import defaultdict
from multiprocessing import Process, Lock, Queue
import traceback
from os.path import isdir

class ChannelValues():
	def __init__(self):
		self.data = {key+1:expand(value) for key, value in enumerate(map(lambda x: float(x)/19, range(20)))}
	def get_random(self):
		return self.data[randint(1,len(self.data))]
	def get_neighbours(self, value, scope = "both"):
		assert scope in ["both", "down", "upper"]
		scope_items = []
		if scope in ["both", "upper"]:
			scope_items.extend([1, 2, 3, 4, 5])
		if scope in ["both", "down"]:
			scope_items.extend([-5, -4, -3, -2, -1])
		assert len(scope_items) > 0
		#print "DEBUG scope items ", scope, scope_items
		for k,v in self.data.iteritems():
			if value == v:
				results = []
				for x in scope_items:
					try:
						results.append(self.data[k + x])
					except:
						pass
				#print "DEBUG2 results: ", results
				return results
		raise ValueError("ColorValues error")

class ArrayImage():
	def __init__(self, image, read_only = False, output_dir = None):
		#importing content of image into 3d array
		self.x = image.shape[0]
		self.y = image.shape[1]
		self.c = 9
		self.read_only = read_only
		self.data = numpy.zeros((self.x, self.y, self.c))
		self.image = image
		self.error_sum = 0
		self.outfile = "output"
		self.outsuffix = ".png"
		self.work_files_counter = 0
		self.final_colors_stat = {}
		self.output_dir = output_dir
		progress_bar = ProgressBar(" importing image ", self.x)
		for x_tmp in range(self.x):
			progress_bar.update(x_tmp)
			for y_tmp in range(self.y):
				rgb_tuple = image[x_tmp, y_tmp]
				self.data[x_tmp][y_tmp][0] = shrink(rgb_tuple[0])#col_obj.lab_l
				self.data[x_tmp][y_tmp][1] = shrink(rgb_tuple[1])#col_obj.lab_a
				self.data[x_tmp][y_tmp][2] = shrink(rgb_tuple[2])#col_obj.lab_b
				self.data[x_tmp][y_tmp][3] = 0#diffs
				self.data[x_tmp][y_tmp][4] = 0
				self.data[x_tmp][y_tmp][5] = 0
				self.data[x_tmp][y_tmp][6] = 0#result
				self.data[x_tmp][y_tmp][7] = 0
				self.data[x_tmp][y_tmp][8] = 0

	def populate_work_statistics(self):
		self.work_colors_stat = {}
		for i, col in enumerate(self.work_colors):
			self.work_colors_stat[i] = col.count
		#print "Final colors stat: ", self.final_colors_stat

	def set_output_filename(self, name):
		if name.endswith('.png') or name.endswith('.jpg'):
			name = name[:-3]
		self.outfile = name
		self.outsuffix = 'png'

	def get_out_folder(self):
		return "{}_{}".format(self.outfile, len(self.work_colors))

	def get_output_filename(self, color_set, work_folder = False):
		bare_name = "{}_{}.{}".format(self.outfile, color_set.count(), self.outsuffix)
		if work_folder == True:
			return "{}/{}".format(self.outfile, bare_name)
		if not self.output_dir is None:
			return "{}/{}".format(self.output_dir, bare_name)
		return bare_name

	def dither(self, color_set, quiet = False):
		color_set.reset_colors_count()

		for x_tmp in range(self.x):
			if quiet == False and x_tmp % 100 == 0:
				print " dithering: ",x_tmp
			for y_tmp in range(self.y):
				self.clip(x_tmp, y_tmp)
				#print self.data[x_tmp][y_tmp][3]
				best_col = self.get_best_color(self.data[x_tmp][y_tmp][0] + self.data[x_tmp][y_tmp][3],
											   self.data[x_tmp][y_tmp][1] + self.data[x_tmp][y_tmp][4],
												self.data[x_tmp][y_tmp][2] + self.data[x_tmp][y_tmp][5],
												color_set)
				self.set_new_color(x_tmp, y_tmp, best_col)
				#if posterizeonly == False:
				self.propagate_errors(x_tmp, y_tmp)
		#print "dithering is over"

	def clip(self, x,y):
		for pos in [3,4,5]:
			#print self.data[x][y][pos]
			max_error = 1
			self.error_sum += abs(self.data[x][y][pos])
			if self.data[x][y][pos] > 1 + max_error:
				self.data[x][y][pos] = 1 + max_error
			if self.data[x][y][pos] < -max_error:
				self.data[x][y][pos] = -max_error

	def propagate_errors(self, x, y):
		r_diff = self.data[x][y][6] - self.data[x][y][0]
		g_diff = self.data[x][y][7] - self.data[x][y][1]
		b_diff = self.data[x][y][8] - self.data[x][y][2]
		propagation_sample= {0: [
			[1,0,7],
			[1,1,1],
			[0,1,5],
			[-1,1,3]
		],
			1: [
			[1,0,1],
			[1,1,1],
			[0,1,1]],
		}
		total_weight = sum([item[2] for item in propagation_sample[(x+y)%2]])
		for item in propagation_sample[(x+y)%2]:
			weight = float(item[2]) / total_weight
			self.push_errors(x + item[0], y + item[1], r_diff * weight, g_diff * weight, b_diff * weight)

	def push_errors(self, x,y, r, g, b):
		if x>=self.x or y >= self.y:
			return
		self.data[x][y][3] -= r
		self.data[x][y][4] -= g
		self.data[x][y][5] -= b
	def get_best_color(self, r,g,b, color_set):
		best_diff = 100000000
		best_col = None
		for col in color_set.iterate():
			cur_diff = get_diff(col.get_small_tuple(), (r,g,b))
			if  cur_diff < best_diff:
				best_diff = cur_diff
				best_col = col
		best_col.count += 1
		return best_col.r, best_col.g, best_col.b

	def set_new_color(self, x, y  , color_tupple):
		self.data[x][y][6] = color_tupple[0]
		self.data[x][y][7] = color_tupple[1]
		self.data[x][y][8] = color_tupple[2]

	def save_new_image(self,color_set, work_folder = False, partial = False):
		if work_folder is True and not os.path.exists(work_image.get_out_folder()):
			os.makedirs(work_image.get_out_folder())
		self.new_image = self.image.copy()

		target_destination = self.get_output_filename(color_set)
		if work_folder is True:
			target_destination = self.get_output_filename(work_folder = True)

		progress_bar = ProgressBar("  @ exporting to {}".format(target_destination), self.x)
		for x in range(0, self.x):
			progress_bar.update(x)
			#if x % 200 == 0:
			#	print " @ exporting {} : {}/{}".format(target_destination, x, self.x)
			for y in range(0, self.y):
				if partial == True and (y < self.y / 10 or y > self.y * 9 / 10):
					self.new_image[x, y, 0] = expand(self.data[x][y][0])
					self.new_image[x, y, 1] = expand(self.data[x][y][1])
					self.new_image[x, y, 2] = expand(self.data[x][y][2])
				else:
					self.new_image[x, y, 0] = expand(self.data[x][y][6])
					self.new_image[x, y, 1] = expand(self.data[x][y][7])
					self.new_image[x, y, 2] = expand(self.data[x][y][8])

		#now exporting color previews
		for pos, color in enumerate(color_set.iterate()):
			cp = ColorPreview(pos, color, self.x)
			for renderdata in cp.render():
				self.new_image[renderdata[0], renderdata[1], 0] = renderdata[2][0]
				self.new_image[renderdata[0], renderdata[1], 1] = renderdata[2][1]
				self.new_image[renderdata[0], renderdata[1], 2] = renderdata[2][2]

		misc.imsave(target_destination, self.new_image)

	def clean_errors(self):
		for x in range(0, self.x):
			for y in range(0, self.y):
				self.data[x, y, 3] = 0
				self.data[x, y, 4] = 0
				self.data[x, y, 5] = 0


class ProgressBar():
	def __init__(self, text, total):
		self.text = "{}".format(text)
		self.state = 0
		self.total = total
	def update(self, current):
		due = current * 10 / self.total
		if due > self.state:
			new_line = '\n'
			if due<9:
				new_line = '\r'
			sys.stdout.write('{} [{:<10}]{}'.format(self.text, '#' * (due+1), new_line))
			sys.stdout.flush()
			self.state = due






def get_diff(col1, col2):
	total = pow(col1[0] - col2[0], 2)# * (1+weights[0])
	total += pow(col1[1] - col2[1], 2)# * (1+weights[1])
	total += pow(col1[2] - col2[2], 2)# * (1+weights[2])
	return total

def shrink(value):
	return pow(value/255.0, 1/2.2)
def expand(value):
	return int(round(pow(value, 2.2) * 255))

class ColorValues():
	def __init__(self, values, big = True):
		if big == True:
			self.R = values[0]
			self.G = values[1]
			self.B = values[2]
			self.recalculate_from_big()
		else:
			self.r = values[0]
			self.g = values[1]
			self.b = values[2]
			self.avg = (self.r + self.g + self.b) / 3
			self.r_diff = self.r - self.avg
			self.g_diff = self.g - self.avg
			self.b_diff = self.b - self.avg
			self.R = expand(self.r)
			self.G = expand(self.g)
			self.B = expand(self.b)
		self.count = 0
	def recalculate_from_big(self):
		self.r = shrink(self.R)
		self.g = shrink(self.G)
		self.b = shrink(self.B)
		self.avg = (self.r + self.g + self.b) / 3
		self.r_diff = self.r - self.avg
		self.g_diff = self.g - self.avg
		self.b_diff = self.b - self.avg
	def copy(self):
		return type(self)(self.R, self.G, self.B)
	def reset_channel(self, channel, value):
		setattr(self, channel, value)
		self.recalculate_from_big()
	def get_small_tuple(self):
		return (self.r, self.g, self.b)

	def get_extended_small_tuple(self):
		return (self.avg, self.r_diff, self.g_diff, self.b_diff)

	def get_big_tuple(self):
		return (self.R, self.G, self.B)

	def __eq__(self, other):
		if isinstance(other, type(self)):
			return self.R == other.R and self.G == other.G and self.B == other.B
		return self.R == other[0] and self.G == other[1] and self.B == other[2]

	def __repr__(self):
		return "{:>03}|{:>03}|{:>03}".format(self.R, self.G, self.B)

class ColorPreview():
	def __init__(self, order, color, x_size):
		self.size = 20
		self.starting_x = x_size - 25 - self.size
		self.starting_y = 25 * (order + 1)
		self.R = color.get_big_tuple()[0]
		self.G = color.get_big_tuple()[1]
		self.B = color.get_big_tuple()[2]
		self.frame_color = (0,0,0)
		if self.R < 50 and self.G < 50 and self.B < 50:
			self.frame_color = (200, 200, 200)
		self.border_thick = self.size / 15
	def render(self):
		for x in range(0, self.size + 1):
			for y in range(0, self.size + 1):
				if x < self.border_thick or self.size - x < self.border_thick or y < self.border_thick or self.size - y < self.border_thick:
					yield self.starting_x + x, self.starting_y + y, self.frame_color
				else:
					yield self.starting_x + x, self.starting_y + y,(self.R, self.G, self.B)

# class MonitoredList(list):
# 	def __init__(self, init_channels, cb):
# 		self.cb = cb
# 		super(MonitoredList, self).__init__(init_channels)
# 	def __setitem__(self, index, value):
# 		self.cb()
# 		super(MonitoredList, self).__setitem__(index, value)
# 	def __iter__(self):
# 		return super(MonitoredList, self).__iter__()

class ColorSet():
	def __init__(self, colors, cv):
		self.colors = colors
		self.cv = cv
		self.verbosity = 0
		#self.original_colors = True
	#def set_non_original(self):
	#	self.original_colors = False
	def __str__(self):
		return "[{}]".format(", ".join([str(col) for col in self.colors]))
	def iterate(self):
		for item in self.colors:
			yield item
	def count(self):
		return len(self.colors)
	def get_lowest_freq_color(self, comparator, tresh = 0):
		comp_str = 'most frequent' if comparator == operator.ge else 'least frequent'
		least_pos = None
		least_freq = None
		for k,v in enumerate(self.colors):
			if least_freq is None or comparator(v.count, least_freq):
				least_freq = v.count
				least_pos = k
		if comparator(least_freq, tresh):
			if self.verbosity > 1:
				print "  returning {} on pos: {}, freq.: {}, tresh: {}".format(comp_str, least_pos, least_freq, tresh)
			return least_pos
		return None

	def print_colors(self, image_size = None, extra_text = "", force = False):
		for i, col in enumerate(self.iterate()):
			if self.verbosity > 0 or force == True:
				if not image_size is None:
					percentage = "{:>5.2f}%".format(100 * float(col.count) / image_size)
				else:
					percentage = ""
				print "  {} {:>2} {:<13} :{:>8}   {}".format(extra_text, i, col, col.count,
																   percentage)

	def mutate(self, img_size):
		if self.verbosity > 0:
			print "  Mutation source: {}".format(self.colors)

		least_frequent_color = self.get_lowest_freq_color(operator.le, tresh=img_size / len(self.colors) / 3)
		most_frequent_color = self.get_lowest_freq_color(operator.ge)

		if not least_frequent_color is None and randint(0,3) == 0:
			#mutatig most frequent into least frequent position
			self.colors[least_frequent_color] \
				= ColorValues(self.mutate_rgb_tupples([self.colors[most_frequent_color].get_big_tuple()])[0])
			print "  1/1 Mutating {}: {}  -> {}: {}".format(most_frequent_color, self.colors[most_frequent_color],
															least_frequent_color, self.colors[least_frequent_color])
		else:
			colors_to_mutate = sample(set(range(len(self.colors))),randint(1,2))
			assert len(colors_to_mutate) in [1,2]
			result = self.mutate_rgb_tupples([self.colors[col].get_big_tuple() for col in colors_to_mutate])
			assert len(colors_to_mutate) == len(result), "Colors to mutate: {}, result: {}".format(colors_to_mutate, result)
			for pos, col in enumerate(colors_to_mutate):
				col_val = ColorValues(result[pos])
				print "  {}/{} Mutating {}: {}  ->  {}".format(pos + 1, len(colors_to_mutate), col, self.colors[col], col_val)
				self.colors[col] = col_val

	def color_exists(self, color):
		for col in self.colors:
			if col == color:
				return True
		return False
	def mutate_rgb_tupples(self, old_rgbs):
		#new_rgb = list(old_rgb)
		src = [0, 1, 2]
		shuffle(src)
		channels_to_mutate = src[:choice([1, 1, 1, 2, 2, 3])]
		new_rgbs = []
		for i,old_rgb in enumerate(old_rgbs):
			#for x in range(20):
			new_rgb = list(old_rgb)
			for ch in channels_to_mutate:

				if len(old_rgbs) == 1:
					second_arg = "both"
				elif i == 0:
					second_arg = "upper"
				else:
					second_arg = "down"

				options = self.cv.get_neighbours(new_rgb[ch], second_arg)
				shuffle(options)
				if len(options) != 0:
					new_rgb[ch] = options

			#print "DEBUG, channels to mutate: {}, rgbs count: {} ".format(channels_to_mutate, len(old_rgbs)),old_rgb, " -> ", new_rgb

			new_rgb = map(lambda x: [x] if isinstance(x, int) else x, new_rgb)
			for possible_rgb in [(r,g,b) for r in new_rgb[0] for g in new_rgb[1] for b in new_rgb[2] ]:
				if not self.color_exists(possible_rgb):
					new_rgbs.append(possible_rgb)
					break
			else:
				print "We dont have good candidate to mutate {}".format(old_rgb)
				new_rgbs.append(old_rgb)
			assert len(new_rgbs) == i + 1 ,"Size of new_rgbs: {}, iteration: {}/{}".format(len(new_rgbs), i + 1, len(old_rgbs))
		return new_rgbs

	def reset_colors_count(self):
		for col in self.colors:
			col.count = 0
	def statistics_exist(self):
		for col in self.colors:
			if col.count > 0:
				return True
		return False

class ColorQueue():
	def __init__(self, color_set_colors_count, lock, cv, size = 5):
		self.queue = []
		self.size = size
		self.color_set_colors_count = color_set_colors_count
		self.lock = lock
		self.cv = cv
	def add(self, colorset, diff):
		if not isinstance(diff, float):
			print "diff must be float"
			sys.exit()
		with self.lock:
			self.queue.append({"diff": diff, "colorset" : copy.deepcopy(colorset)})
			self.queue = sorted(self.queue, key=lambda k: k['diff'])
			self.queue = self.queue[:self.size]
	def __str__(self):
		return "Queue: {}".format("; ".join(["{:.5f}: {}".format(item["diff"], str(item["colorset"])) for item in self.queue]))
	def get_best_diff(self):
		if len(self.queue) == 0:
			return  1000000
		return self.queue[0]["diff"]
	def get(self, position = 0):
		if len(self.queue) < min(self.size, 3):
			print "Size of Color_queue is below treshold, generating random colors"
			return ColorSet(self.get_random_colors(self.color_set_colors_count), self.cv)
		if position >= len(self.queue):
			print "   Position {} > length of queue: {}, returning position 0".format(position, len(self.queue))
			position = 0
		new_color_set = copy.deepcopy(self.queue[position]["colorset"])
		#for item in new_color_set.iterate():
		#	item.count = 0
		return new_color_set
	def pretty_print(self):
		with self.lock:
			rows_to_print = ["QUEUE:"]
			for col in self.queue:
				rows_to_print.append("{:.4f}: {}".format(col["diff"], str(col["colorset"])))
			for row in rows_to_print:
				print "  > {}".format(row)
	def get_random_colors(self, colors_count):
		#print "genrating {} colors".format(colors_count)
		random_colors = []
		for x in range(colors_count):
			best_dist = -1
			distant_color = None
			for y in range(10):
				col = self.get_random_color(random_colors)
				#print "considering color: {}".format(col)
				dist = get_nearest_distance(col, random_colors)
				if dist > best_dist:
					best_dist = dist
					distant_color = col
			if distant_color is not None:
				random_colors.append(distant_color)
		print " Generated colors: {}".format(random_colors)
		return random_colors

	def get_random_single_color(self):
		return self.cv.get_random()
		#return choice([0, 0, 10, 25, 60, 140, 190, 255, 255])

	def get_random_color(self, current_colors = None):
		for x in range(100):
			col_tmp = ColorValues([self.get_random_single_color(), self.get_random_single_color(), self.get_random_single_color()])
			if current_colors is None or col_tmp not in current_colors:
				return col_tmp
		raise ValueError("failed to generate color")


def get_args():
	'''This function parses and return arguments passed in'''
	# Assign description to the help doc
	parser = argparse.ArgumentParser(
		description="image ditherer")

	# Add arguments

	parser.add_argument(
		'-c', '--colors', type=int, default = 5)
	parser.add_argument(
		'-t', '--threads', type=int, default = 2)
	parser.add_argument(
		'-p', '--percentage', type=int, default = 100)
	parser.add_argument(
		'-i', '--idleiterations', type=int, default = 75)
	parser.add_argument(
		'-o', '--outfile', type=str, default="output")
	parser.add_argument(
		'-d', '--outdir', help ="target directory for output files", type=str, default="dithered_images")
	parser.add_argument(
		'-s', '--saveworkimages', action = "store_true", default = False)
	parser.add_argument(
		'-A', '--partial', action = "store_true", default = False)
	parser.add_argument(
		'-f', '--infile', required = True, type=str, nargs='+')
	parser.add_argument("-v", "--verbosity", action="count",
						help="increase output verbosity", default = 0)
	#parser.add_argument("-P", "--posterizeonly", action = "store_true", default = False)


	args = parser.parse_args()
	colors = args.colors
	percentage = args.percentage
	outfile = args.outfile
	infile = args.infile
	idleiterations = args.idleiterations
	#posterizeonly = args.posterizeonly
	saveworkimages = args.saveworkimages
	partial = args.partial
	threads = args.threads
	outdir = args.outdir
	verbosity = args.verbosity

	return percentage, colors, outfile, infile, idleiterations, saveworkimages, partial, threads, outdir, verbosity

def get_nearest_distance(col1, colors):
	best_diff = 0
	for col in colors:
		diff = get_diff(col1.get_small_tuple(), col.get_small_tuple())
		if diff > best_diff:
			best_diff = diff
	return best_diff





def run(color_queue, work_image, last_change, x, save_lock, thread_id, q, cv, partial, saveworkimages, verbosity):
	# Takes:
	# copy of workimage
	# copy of colorset
	# returns:
	# modified colorset
	# modified workimage


	try:
		ColorSet.verbosity = verbosity
		if verbosity > 0:
			with save_lock:
				print " * {:>3}/{}".format(x, thread_id)
		elif thread_id == 0:
			print " * iteration {:>3}".format(x)
		# preparing for iteration
		queue_pos = randint(0,3)
		color_set = color_queue.get(queue_pos)
		color_set.verbosity = verbosity

		#if color_set == False: # no mutation freshely generated colors
		if color_set.statistics_exist():
			color_set.mutate(work_image.x * work_image.y)
		else:
			if verbosity > 0:
				print "  Do not mutating..."
		action = "mutate"



		work_image.error_sum = 0
		work_image.clean_errors()
		work_image.dither(color_set, quiet = True)
		#least_frequency = 100000
		changed = False

		with save_lock:
			color_set.print_colors(work_image.x * work_image.y, extra_text = " TH:{}".format(thread_id))
			current_error = float(work_image.error_sum) / work_image.x / work_image.y
			print "  TH:{} Actual error: {:.04}% (best: {:.04f}, last change: {:>2} ago, queue pos: {})".\
				format(thread_id, current_error, color_queue.get_best_diff(), x - last_change, queue_pos)
			if current_error < color_queue.get_best_diff():
				work_image.save_new_image(color_set, partial = partial)
				if saveworkimages == True:
					work_image.save_new_image(color_set, work_folder = True, partial = partial)
					work_image.work_files_counter += 1
				changed = True

		q.put((color_set, current_error, changed))
	except Exception as e:
		with save_lock:
			print "Run() failed with: ",str(e)
			traceback.print_exc()
			q.put(e)
	except KeyboardInterrupt:
		q.put(False)



if __name__ == "__main__":

	percentage, colors, outfile, infiles, idleiterations, saveworkimages, partial, threads, outdir, verbosity = get_args()

	if not isdir(outdir):
		try:
			os.mkdir(outdir)
		except Exception as e:
			print "ERROR: {} not created: {}".format(outdir, str(e))
			sys.exit()

	#print "Posterizing only: ", posterizeonly

	ColorSet.verbosity = verbosity

	max_error = 0.5

	#test
	test_v = randint (10,250)
	print " {} -> {} -> {}".format(test_v, shrink(test_v), expand(shrink(test_v)))
	if not test_v == expand(shrink(test_v)):
		sys.exit()

	cv = ChannelValues()
	save_lock = Lock()
	q = Queue()

	for file_tmp in infiles:

		inimage = misc.imread(file_tmp)
		print "Doing: {} ({}x{}px)".format(file_tmp, inimage.shape[0], inimage.shape[1])
		if percentage != 100:
			inimage = misc.imresize(inimage, percentage)
			print " ... resizing to {}%, new dimensions: {}x{}".format(percentage, inimage.shape[0], inimage.shape[1])

		work_image = ArrayImage(inimage, output_dir = outdir)
		work_image.set_output_filename(file_tmp.rsplit('.',1)[0])

		color_queue = ColorQueue(colors, Lock(), cv)

		#print " Clipped amount: {}".format(work_image.error_sum)

		last_change = 0
		action = "initial"
		action_results = defaultdict(int)
		queue_pos = -1


		for x in range(2000):

			procs = [Process(target=run, args=(color_queue, copy.deepcopy(work_image), last_change, x,
											   save_lock, i, q, cv, partial, saveworkimages, verbosity)) for i in range(threads)]
			for p in procs:
				p.start()
			for p in procs:
				result = q.get()
				if isinstance(result, Exception) or result is False:
					with save_lock:
						print "Thread execution failed...."
					sys.exit()
				color_set, current_error, changed = result
				#print "From queue: ", color_set, current_error
				p.join()
				with save_lock:
					color_queue.add(color_set, current_error)
					if changed == True:
						last_change = x
			color_queue.pretty_print()

			if x - last_change > idleiterations / threads:
				print " * processing of {} done ....".format(file_tmp)
				break


		print action_results



