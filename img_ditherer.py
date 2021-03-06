from scipy import misc
import numpy
import argparse
#from itertools import combinations_with_replacement, permutations, product
#from colormath.color_diff import delta_e_cie2000
from random import randint, choice, shuffle, sample
import sys
import copy
import os
import operator
#from collections import defaultdict
from multiprocessing import Process, Lock, Queue
import traceback
from os.path import isdir

class Runner(object):
	def __init__(self, image_file, color_queue, partial, required_percentage, megapixels):
		self.tmp_file = image_file
		self.partial = partial
		self.color_queue = color_queue
		self.pass_step = 0
		if megapixels is None:
			self.percentage = required_percentage / 100.0
		else:
			inimage = misc.imread(self.tmp_file)
			orig_size = inimage.shape[0] * inimage.shape[1]
			self.percentage = pow(1000000 * megapixels / orig_size, 0.5)
			print "Converting image with size {} target megapixels {} to {} percentage".format(orig_size, megapixels, self.percentage)
		assert self.percentage > 0.05
		assert self.percentage < 3

	def generate_dithered_img(self, relative_percentage):
		work_image = self.get_work_image(relative_percentage * self.percentage)
		work_image.dither(self.color_queue.get(), quiet=True)
		work_image.set_output_filename(self.tmp_file.rsplit('.', 1)[0])
		work_image.save_new_image(self.color_queue.get(), partial=self.partial)

	def get_work_image(self, percentage):
		self.inimage = misc.imread(self.tmp_file)
		print "Doing: {} pass: {} ({}x{}px)".format(self.tmp_file, self.pass_step, self.inimage.shape[0], self.inimage.shape[1])
		if percentage != 100:
			self.inimage = misc.imresize(self.inimage, percentage)
			print " ... resizing to {}%, new dimensions: {}x{}".format(percentage, self.inimage.shape[0], self.inimage.shape[1])
		return ArrayImage(self.inimage, output_dir = outdir)

	def run(self, relative_percentage, iterations, diff_malus):
		self.pass_step += 1
		final_percentage = self.percentage * relative_percentage
		assert final_percentage >= 0.01
		work_image = self.get_work_image(self.percentage * relative_percentage)
		work_image.set_output_filename(file_tmp.rsplit('.',1)[0])

		last_change = 0
		#action = "initial"
		queue_pos = -1
		diff_stat = []

		for x in range(iterations):

			procs = [Process(target=run, args=(color_queue, copy.deepcopy(work_image), last_change, x,
											   save_lock, i, q, cv, self.partial, saveworkimages, verbosity, self.pass_step)) for i in range(threads)]
			for p in procs:
				p.start()
			for p in procs:
				result = q.get()
				if isinstance(result, Exception) or result is False:
					with save_lock:
						print "Thread execution failed...."
					sys.exit()
				color_set, current_error, changed = result
				current_error *= diff_malus
				#print "From queue: ", color_set, current_error
				p.join()
				with save_lock:
					self.color_queue.add(color_set, current_error)
					if changed == True:
						last_change = x
			self.color_queue.pretty_print()
			diff_stat.append(self.color_queue.get_best_diff())

			if x - last_change > idleiterations / threads:
				print " * processing of {} done ....".format(file_tmp)
				break

class ChannelValues():
	def __init__(self, grades = 20):
		self.data = {key+1:expand(value) for key, value in enumerate(map(lambda x: float(x)/(grades - 1), range(grades)))}
	def get_random(self):
		return self.data[randint(1,len(self.data))] #dictionary starts with key 1
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
	def __init__(self, image, output_dir = None):
		#importing content of image into 3d array
		self.x = image.shape[0]
		self.y = image.shape[1]
		self.c = 9
		#self.read_only = read_only
		self.data = numpy.zeros((self.x, self.y, self.c))
		self.image = image
		self.error_sum = 0
		self.outfile = "output"
		self.outsuffix = ".png"
		#self.work_files_counter = 0
		#self.final_colors_stat = {}
		self.output_dir = output_dir
		progress_bar = ProgressBar(" importing image ", self.x)
		for x_tmp in range(self.x):
			progress_bar.update(x_tmp)
			for y_tmp in range(self.y):
				rgb_tuple = image[x_tmp, y_tmp]
				self.data[x_tmp][y_tmp][0] = shrink(rgb_tuple[0])#col_obj.lab_l
				self.data[x_tmp][y_tmp][1] = shrink(rgb_tuple[1])#col_obj.lab_a
				self.data[x_tmp][y_tmp][2] = shrink(rgb_tuple[2])#col_obj.lab_b
				self.data[x_tmp][y_tmp][3] = 0#diffs, first the leftover from previously processed, then actual
				self.data[x_tmp][y_tmp][4] = 0
				self.data[x_tmp][y_tmp][5] = 0
				self.data[x_tmp][y_tmp][6] = 0#result
				self.data[x_tmp][y_tmp][7] = 0
				self.data[x_tmp][y_tmp][8] = 0

	def set_output_filename(self, name):
		if name.endswith('.png') or name.endswith('.jpg'):
			name = name[:-3]
		self.outfile = name
		self.outsuffix = 'png'

	def get_output_dir(self, color_set, work_folder = False):
		outdir = []
		if not self.output_dir is None:
			outdir.append("{}".format(self.output_dir))
		if work_folder == True:
			outdir.append("{}_{}".format(self.outfile, color_set.count()))
		return '/'.join(outdir)

	def get_output_filename(self, color_set, work_folder = False, iteration = None, pass_step = None):
		iteration_str = ""
		if not iteration is None:
			iteration_str = '_{:>03}'.format(iteration)
		pass_str = ""
		if not pass_step is None:
			pass_str = '_{}'.format(pass_step)
		bare_name = "{}_{}{}{}.{}".format(self.outfile, color_set.count(), pass_str, iteration_str, self.outsuffix)
		return '/'.join([self.get_output_dir(color_set, work_folder), bare_name])

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
				self.propagate_errors(x_tmp, y_tmp)
				self.set_posprocess_diff(x_tmp, y_tmp)

	def set_posprocess_diff(self, x, y):
		for pos in [3, 4, 5]:
			self.data[x][y][pos] = self.data[x][y][pos + 3] - self.data[x][y][pos - 3]

	def get_local_error(self, x, y, pos):
		current_error = abs(self.data[x][y][pos])
		square_error = 0
		for x_tmp in [x, x+1]:
			for y_tmp in [y, y+1]:
				try:
					square_error += self.data[x_tmp][y_tmp][pos]
				except:
					pass
		square_error = abs(square_error)
		#if square_error < current_error:
		#	print "ERRORS: {} vs {}".format(current_error, square_error)
		return min(current_error, square_error)


	def clip(self, x,y):
		for pos in [3,4,5]:
			#print self.data[x][y][pos]
			max_error = 3
			self.error_sum +=self.get_local_error(x,y,pos)
			#self.error_sum += abs(self.data[x][y][pos])
			if self.data[x][y][pos] > 1 + max_error:
				print "Clipping: {}".format(self.data[x][y][pos] - 1 - max_error)
				self.data[x][y][pos] = 1 + max_error
			if self.data[x][y][pos] < -max_error:
				print "Clipping: {}".format(self.data[x][y][pos] + max_error)
				self.data[x][y][pos] = -max_error

	def propagate_errors(self, x, y):
		r_diff = self.data[x][y][6] - self.data[x][y][0]
		g_diff = self.data[x][y][7] - self.data[x][y][1]
		b_diff = self.data[x][y][8] - self.data[x][y][2]
		propagation_sample= {0: [
			[1,0,7],
			[1,1,1],
			[0,1,5],
			[-1,1,3]],
			1: [
			[1,0,1],
			[1,1,1],
			[0,1,1]],
			2: [
			[1,0,5],
			[1,1,1],
			[0,1,4],
			[-1,1,4],
			[-2,1,4]],
			3: [
			[1, 0, 3],
			[1, 1, 1],
			[0, 1, 1],
			[-1, 1, 1]]
		}
		total_weight = sum([item[2] for item in propagation_sample[(x+y)%len(propagation_sample)]])
		for item in propagation_sample[(x+y)%len(propagation_sample)]:
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

	def save_new_image(self, color_set, work_folder = False, partial = False, iteration = None, pass_step = None):
		self.new_image = self.image.copy()
		target_path = self.get_output_dir(color_set, work_folder=work_folder)
		target_file_with_path = self.get_output_filename(color_set, work_folder=work_folder, iteration = iteration,
														 pass_step = pass_step)

		if not os.path.exists(target_path):
			os.makedirs(target_path)

		progress_bar = ProgressBar("  @ exporting to {}".format(target_file_with_path), self.x)
		for x in range(0, self.x):
			progress_bar.update(x)
			for y in range(0, self.y):
				if partial == True:
					break1 = self.y / 10
					break2 = self.y * 9 / 10
					if (y < break1 or y > break2):
						self.new_image[x, y, 0] = expand(self.data[x][y][0])
						self.new_image[x, y, 1] = expand(self.data[x][y][1])
						self.new_image[x, y, 2] = expand(self.data[x][y][2])
						continue
					elif y == break1 or y == break2:
						self.new_image[x, y, 0] = 0
						self.new_image[x, y, 1] = 0
						self.new_image[x, y, 2] = 0
						continue
				self.new_image[x, y, 0] = expand(self.data[x][y][6])
				self.new_image[x, y, 1] = expand(self.data[x][y][7])
				self.new_image[x, y, 2] = expand(self.data[x][y][8])

		#now exporting color previews
		for pos, color in enumerate(color_set.iterate()):
			#print "DEBUG ",color_set.count()
			cp = ColorPreview(pos, color, self.x, color_set.count())
			for renderdata in cp.render():
				self.new_image[renderdata[0], renderdata[1], 0] = renderdata[2][0]
				self.new_image[renderdata[0], renderdata[1], 1] = renderdata[2][1]
				self.new_image[renderdata[0], renderdata[1], 2] = renderdata[2][2]

		misc.imsave(target_file_with_path, self.new_image)

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
	assert len(col1) == len(col2)
	if len(col1) == 3:
		total = pow(col1[0] - col2[0], 2)# * (1+weights[0])
		total += pow(col1[1] - col2[1], 2)# * (1+weights[1])
		total += pow(col1[2] - col2[2], 2)# * (1+weights[2])
		return total
	assert len(col1) == 4
	#so we got extended tuple where first item is avg (~brightness)
	total = pow((col1[0] - col2[0]) / 2, 2)
	total += pow(col1[1] - col2[1], 2)
	total += pow(col1[2] - col2[2], 2)
	total += pow(col1[3] - col2[3], 2)
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
	def __init__(self, order, color, x_size, color_count):
		self.size = min(25, x_size / color_count / 2)
		self.colors_count = color_count
		self.starting_x = x_size - 25 - self.size
		self.starting_y = (self.size + 5) * (order + 1)
		#print "DEBUG ", self.starting_x, self.starting_y
		self.R = color.get_big_tuple()[0]
		self.G = color.get_big_tuple()[1]
		self.B = color.get_big_tuple()[2]
		self.frame_color = (0,0,0)
		if self.R < 80 and self.G < 80 and self.B < 80:
			self.frame_color = (200, 200, 200)
		self.border_thick = self.size / 15
	def render(self):
		for x in range(0, self.size + 1):
			for y in range(0, self.size + 1):
				if x < self.border_thick or self.size - x < self.border_thick or y < self.border_thick or self.size - y < self.border_thick:
					yield self.starting_x + x, self.starting_y + y, self.frame_color
				else:
					yield self.starting_x + x, self.starting_y + y,(self.R, self.G, self.B)


class ColorSet():
	def __init__(self, colors, cv, min_sat):
		self.colors = colors
		self.cv = cv
		self.verbosity = 0
		self.min_sat = min_sat

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

	def get_saturations(self, color):
		max_value = shrink(max(color))
		min_value = shrink(min(color))
		return max_value - min_value

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
				if not self.color_exists(possible_rgb) and self.get_saturations(possible_rgb) > self.min_sat:
					new_rgbs.append(possible_rgb)
					break
			else:
				print "   We dont have good candidate to mutate {}".format(old_rgb)
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
	def __init__(self, color_set_colors_count, lock, min_sat, cv, size = 5):
		self.queue = []
		self.size = size
		self.color_set_colors_count = color_set_colors_count
		self.lock = lock
		self.cv = cv
		self.min_sat = min_sat
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
			return ColorSet(self.get_random_colors(self.color_set_colors_count), self.cv, self.min_sat)
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
		description="Image ditherer, looking for optimal unique colors so that errors are as minimal as possible")

	# Add arguments

	parser.add_argument(
		'-c', '--colors', type=int, help = "unique colors for final image", default = 5)
	parser.add_argument(
		'-t', '--threads', type=int, help = "threads count", default = 2)
	group = parser.add_mutually_exclusive_group()
	group.add_argument(
		'-p', '--percentage', type=int, help = "for resizing the image, default 100. "
											   "Smaller images are processed faster of course", default = 100)
	group.add_argument(
		'-M', '--megapixels', type=float, help = "size of final image", default = None)
	parser.add_argument(
		'-i', '--idleiterations', type=int, help = "The script terminates if n iterations brought no improvement."
												   " Default = 75. Actually it waits iterations/threads iterations.", default = 75)
	parser.add_argument(
		'-o', '--outfile', type=str, help = "does not work for now", default="output")
	parser.add_argument(
		'-d', '--outdir', help ="target directory for output files", type=str, default="dithered_images")
	parser.add_argument(
		'-s', '--saveworkimages', action = "store_true", help = "save work images "
																"(after iteration that brough improvement)", default = False)
	parser.add_argument(
		'-A', '--partial', action = "store_true", help = "Leave strips of original image on the sides for comparison",default = False)
	parser.add_argument(
		'-f', '--infile', required = True, help = "one of more images that will be processed", type=str, nargs='+')
	parser.add_argument("-v", "--verbosity", action="count",
						help="increase output verbosity", default = 0)
	parser.add_argument("-m", "--minsat", type = float, default = 0, help = "minimal saturation")

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
	minsat = args.minsat
	megapixels = args.megapixels

	return percentage, colors, outfile, infile, idleiterations,\
		   saveworkimages, partial, threads, outdir, verbosity, minsat, megapixels

def get_nearest_distance(col1, colors):
	best_diff = 0
	for col in colors:
		#diff = get_diff(col1.get_small_tuple(), col.get_small_tuple())
		diff = get_diff(col1.get_extended_small_tuple(), col.get_extended_small_tuple())
		if diff > best_diff:
			best_diff = diff
	return best_diff


def run(color_queue, work_image, last_change, x, save_lock, thread_id, q, cv, partial, saveworkimages, verbosity, pass_step):
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
			print " * iteration {:>3} in pass {}, last change {} iterations ago".format(x, pass_step, x - last_change)
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

		work_image.error_sum = 0
		work_image.clean_errors()
		work_image.dither(color_set, quiet = True)
		#least_frequency = 100000
		changed = False

		with save_lock:
			color_set.print_colors(work_image.x * work_image.y, extra_text = " TH:{}".format(thread_id))
			current_error = float(work_image.error_sum) / work_image.x / work_image.y
			print "  TH:{} Actual error: {:.4}% ({:+.4f}, parent queue pos: {})".\
				format(thread_id, current_error, current_error - color_queue.get_best_diff(), queue_pos)
			if current_error < color_queue.get_best_diff():
				work_image.save_new_image(color_set, partial = partial)
				if saveworkimages == True:
					work_image.save_new_image(color_set, work_folder = True, partial = partial,
											  iteration = x, pass_step = pass_step)
					#work_image.work_files_counter += 1
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

	required_percentage, colors, outfile, infiles, idleiterations, saveworkimages,\
	partial, threads, outdir, verbosity, min_sat, megapixels = get_args()

	if min_sat > 0.2:
		min_sat = 0.2
	print "Minimal saturation: {}".format(min_sat)

	if not isdir(outdir):
		try:
			os.mkdir(outdir)
		except Exception as e:
			print "ERROR: {} not created: {}".format(outdir, str(e))
			sys.exit()

	#print "Posterizing only: ", posterizeonly

	ColorSet.verbosity = verbosity

	#test
	test_v = randint (10,250)
	#print " {} -> {} -> {}".format(test_v, shrink(test_v), expand(shrink(test_v)))
	assert test_v == expand(shrink(test_v))

	cv = ChannelValues()
	save_lock = Lock()
	q = Queue()

	for file_tmp in infiles:

		color_queue = ColorQueue(colors, Lock(), min_sat, cv)
		runner = Runner(file_tmp, color_queue, partial, required_percentage, megapixels)
		runner.run(0.25, 20, 1.1)
		runner.run(0.5, 20, 1.05)
		runner.generate_dithered_img(1)
		runner.run(1, 1000, 1)







