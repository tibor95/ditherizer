from scipy import misc
import numpy
import argparse
from itertools import combinations_with_replacement, permutations, product
from colormath.color_diff import delta_e_cie2000
from random import randint, choice, shuffle
import sys
import copy
import os
import operator

class ChannelValues():
    def __init__(self):
        self.data = {key+1:expand(value) for key, value in enumerate(map(lambda x: float(x)/14, range(15)))}
    def get_random(self):
        return self.data[randint(1,len(self.data))]
    def get_neighbours(self, value):
        for k,v in self.data.iteritems():
            if value == v:
                results = []
                for x in [-3, -2, -1, 1, 2, 3]:
                    try:
                        results.append(self.data[k + x])
                    except:
                        pass

                return results
        raise ValueError("ColorValues error")

class ArrayImage():
    def __init__(self, image, read_only = False):
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
        for x_tmp in range(self.x):
            if x_tmp % 100 == 0:
                print " importing: ",x_tmp
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
        #print "image imported to the array"

    def reset_colors_count(self):
        for col in self.work_colors:
            col.count = 0
        #for col in self.final_colors:
        #    col.count = 0
    def populate_final_statistics(self):
        self.final_colors_stat = {}
        for i, col in enumerate(self.final_colors):
            self.final_colors_stat[i] = col.count

    def populate_work_statistics(self):
        self.work_colors_stat = {}
        for i, col in enumerate(self.work_colors):
            self.work_colors_stat[i] = col.count
        #print "Final colors stat: ", self.final_colors_stat
    def get_lowest_freq_color(self, comparator, tresh = 0):
        comp_str = 'most frequent' if comparator == operator.ge else 'least frequent'
        self.populate_work_statistics()
        least_pos = None
        least_freq = None
        for k,v in self.work_colors_stat.iteritems():
            if least_freq is None or comparator(v, least_freq):
                #print "  DEBUG using {}  {}".format(k, v)
                least_freq = v
                least_pos = k
            #else:
                #print "  DEBUG not using {}  {}".format(k, v)
        #print "   considering {} on  pos: {}, freq.: {}, tresh: {}".format(comp_str, least_pos, least_freq, tresh)
        if comparator(least_freq, tresh):
            #print "  returning {} on pos: {}, freq.: {}, tresh: {}".format(comp_str, least_pos, least_freq, tresh)
            return least_pos
        return None
    def set_output_filename(self, name):
        if name.endswith('.png') or name.endswith('.jpg'):
            name = name[:-3]
        self.outfile = name
        self.outsuffix = 'png'
    def get_out_folder(self):
        return "{}_{}".format(self.outfile, len(self.work_colors))
    def get_output_filename(self, work_folder = False):
        if work_folder == False:
            return "{}_{}.{}".format(self.outfile, len(self.work_colors), self.outsuffix)
        return "{}/{}_{}_{:>03}.{}".format(self.get_out_folder(), self.outfile, len(self.work_colors), self.work_files_counter, self.outsuffix)
    def set_colors_to_use(self, colors):
        self.work_colors = colors
        self.final_colors = copy.deepcopy(self.work_colors)
        #print self.colors
    def colors_to_final(self):
        self.final_colors = copy.deepcopy(self.work_colors)
        #print "DEBUG count transfer to final ", id(self.final_colors), self.final_colors[0].count, id(self.work_colors), self.work_colors[0].count
    def final_colors_to_work(self):
        self.work_colors = copy.deepcopy(self.final_colors)
        #print "DEBUG count transfer ",id(self.final_colors), self.final_colors[0].count, id(self.work_colors), self.work_colors[0].count
    def mutate_rgb_tupple(self, old_rgb):
        new_rgb = list(old_rgb)
        src = [0,1,2]
        shuffle(src)
        #choice([1, 1, 1, 2, 2, 3])
        channels_to_mutate = src[:choice([1,1,1,2,2,3])]
        for ch in channels_to_mutate:
            new_rgb[ch] = choice(cv.get_neighbours(new_rgb[ch]))
        return tuple(new_rgb)
    def mutate(self):
        print "  Mutation source: {}".format(self.final_colors)
        self.work_colors = copy.deepcopy(self.final_colors)

        preferred_candidate = self.get_lowest_freq_color(operator.le, tresh=self.x * self.y / len(self.final_colors) / 3)
        #self.get_lowest_freq_color(operator.ge, tresh=self.x * self.y / len(self.final_colors) / 3)

        if preferred_candidate is None or randint(0,3) == 0:
            #mutating random color
            col_to_mutate = randint(0, len(self.final_colors) - 1)
            #channel_to_mutate = randint(0, 2)

            rgb = self.final_colors[col_to_mutate].get_big_tuple()
            new_rgb = self.mutate_rgb_tupple(rgb)
            print " Mutating {}: {}  ->  {}".format(col_to_mutate, rgb, new_rgb)
            self.work_colors[col_to_mutate] = ColorValues(new_rgb)
        else:
            #mutating existing color
            if randint(0,1) == 0:
                source_color = self.get_lowest_freq_color(operator.ge)
                if source_color == preferred_candidate:
                    print "ERROR: {} == {}".format(str(source_color), str(preferred_candidate))
                    sys.exit()
            else:
                source_color = preferred_candidate

            rgb = self.final_colors[source_color].get_big_tuple()
            new_rgb = self.mutate_rgb_tupple(rgb)
            if source_color != preferred_candidate:
                print " Mutating {} -> {}: {}  ->  {}".format(source_color, preferred_candidate, rgb, new_rgb)
            else:
                print " Mutating {}: {}  ->  {}".format(preferred_candidate, rgb, new_rgb)
            self.work_colors[preferred_candidate] = ColorValues(new_rgb)
        #do we have natural candidate?



    def dither(self, quiet = False):
        self.reset_colors_count()

        for x_tmp in range(self.x):
            if quiet == False and x_tmp % 100 == 0:
                print " dithering: ",x_tmp
            for y_tmp in range(self.y):
                self.clip(x_tmp, y_tmp)
                #print self.data[x_tmp][y_tmp][3]
                best_col = self.get_best_color(self.data[x_tmp][y_tmp][0] + self.data[x_tmp][y_tmp][3],
                                               self.data[x_tmp][y_tmp][1] + self.data[x_tmp][y_tmp][4],
                                                self.data[x_tmp][y_tmp][2] + self.data[x_tmp][y_tmp][5])
                self.set_new_color(x_tmp, y_tmp, best_col)
                if posterizeonly == False:
                    self.propagate_errors(x_tmp, y_tmp)
        #print "dithering is over"
    def clip(self, x,y):
        for pos in [3,4,5]:
            #print self.data[x][y][pos]
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
        # 2: [
        #     [1,0,6],
        #     [1,1,1],
        #     [0,1,4],
        #     [-1,1,3]
        #],
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
    def get_best_color(self, r,g,b ):
        #avg = (r + g + b) / 3
        #r_diff = avg - r
        #g_diff = avg - g
        #b_diff = avg - b
        best_diff = 100000000
        best_col = None
        #lab_color = LabColor(l,a,b)
        for col in self.work_colors:
            cur_diff = self.get_diff(col.get_small_tuple(), (r,g,b))
            if  cur_diff < best_diff:
                best_diff = cur_diff
                best_col = col
        best_col.count += 1
        return best_col.r, best_col.g, best_col.b
    def get_diff(self, col1, col2):
        #print type(col1), type(col2)
        total = pow(col1[0] - col2[0], 2)# * (1+weights[0])
        total += pow(col1[1] - col2[1], 2)# * (1+weights[1])
        total += pow(col1[2] - col2[2], 2)# * (1+weights[2])
        return total
    def set_new_color(self, x, y  , color_tupple):
        self.data[x][y][6] = color_tupple[0]
        self.data[x][y][7] = color_tupple[1]
        self.data[x][y][8] = color_tupple[2]
    def save_new_image(self,work_folder = False, partial = False):
        if work_folder is True and not os.path.exists(work_image.get_out_folder()):
            os.makedirs(work_image.get_out_folder())
        self.new_image = self.image.copy()

        target_destination = self.get_output_filename()
        if work_folder is True:
            target_destination = self.get_output_filename(work_folder = True)

        for x in range(0, self.x):
            if x % 200 == 0:
                print " @ exporting {} : {}".format(target_destination, x)
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
        for pos, color in enumerate(self.work_colors):
            cp = ColorPreview(pos, color, work_image.x)
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
            self.r = shrink(self.R)
            self.g = shrink(self.G)
            self.b = shrink(self.B)
            self.avg = (self.r + self.g + self.b) / 3
            self.r_diff = self.r - self.avg
            self.g_diff = self.g - self.avg
            self.b_diff = self.b - self.avg
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
    def get_small_tuple(self):
        return (self.r, self.g, self.b)

    def get_extended_small_tuple(self):
        return (self.avg, self.r_diff, self.g_diff, self.b_diff)

    def get_big_tuple(self):
        return (self.R, self.G, self.B)

    def __eq__(self, other):
        return self.R == other.R and self.G == other.G and self.B == other.B

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
        self.border_thick = self.size / 15
    def render(self):
        for x in range(0, self.size + 1):
            for y in range(0, self.size + 1):
                if x < self.border_thick or self.size - x < self.border_thick or y < self.border_thick or self.size - y < self.border_thick:
                    yield self.starting_x + x, self.starting_y + y,(0,0,0)
                else:
                    yield self.starting_x + x, self.starting_y + y,(self.R, self.G, self.B)




def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description="image ditherer")

    # Add arguments

    parser.add_argument(
        '-c', '--colors', type=int, default = 5)
    parser.add_argument(
        '-p', '--percentage', type=int, default = 100)
    parser.add_argument(
        '-i', '--idleiterations', type=int, default = 100)
    parser.add_argument(
        '-o', '--outfile', type=str, default="output")
    parser.add_argument(
        '-s', '--saveworkimages', action = "store_true", default = False)
    parser.add_argument(
        '-A', '--partial', action = "store_true", default = False)
    parser.add_argument(
        '-f', '--infile', required = True, type=str, nargs='+')
    parser.add_argument("-P", "--posterizeonly", action = "store_true", default = False)


    args = parser.parse_args()
    colors = args.colors
    percentage = args.percentage
    outfile = args.outfile
    infile = args.infile
    idleiterations = args.idleiterations
    posterizeonly = args.posterizeonly
    saveworkimages = args.saveworkimages
    partial = args.partial

    return percentage, colors, outfile, infile, posterizeonly, idleiterations, saveworkimages, partial

def get_nearest_distance(col1, colors):
    best_diff = 0
    for col in colors:
        diff = work_image.get_diff(col1.get_small_tuple(), col.get_small_tuple())
        if diff > best_diff:
            best_diff = diff
    return best_diff

def get_random_single_color():
    return cv.get_random()
    #return choice([0, 0, 10, 25, 60, 140, 190, 255, 255])

def get_random_color(current_colors = None):
    for x in range(100):
        col_tmp = ColorValues([get_random_single_color(), get_random_single_color(), get_random_single_color()])
        if current_colors is None or col_tmp not in current_colors:
            return col_tmp
    raise ValueError("failed to generate color")

def get_random_colors(colors_count):
    #print "genrating {} colors".format(colors_count)
    random_colors = []
    for x in range(colors_count):
        best_dist = -1
        distant_color = None
        #col = get_random_color()
        for y in range(10):
            col = get_random_color(random_colors)
            #print "considering color: {}".format(col)
            dist = get_nearest_distance(col, random_colors)
            # print dist
            if dist > best_dist:
                best_dist = dist
                distant_color = col
                 # print "better: {}".format(best_dist)
        if distant_color is not None:
            #print "Appending: {}".format(distant_color)
            random_colors.append(distant_color)
        #random_colors.append(col)
    print " Generated colors: {}".format(random_colors)
    return random_colors

percentage, colors, outfile, infiles, posterizeonly, idleiterations, saveworkimages, partial = get_args()

print "Posterizing only: ", posterizeonly

max_error = 0.5

#test
test_v = randint (10,250)
print " {} -> {} -> {}".format(test_v, shrink(test_v), expand(shrink(test_v)))
if not test_v == expand(shrink(test_v)):
    sys.exit()

cv = ChannelValues()

for file_tmp in infiles:

    print "Doing: {}".format(file_tmp)

    inimage = misc.imread(file_tmp)
    if percentage != 100:
        inimage = misc.imresize(inimage, percentage)
        print " ... resizing to {}%, new dimensions: {}x{}".format(percentage, inimage.shape[0], inimage.shape[1])

    work_image = ArrayImage(inimage)
    work_image.set_output_filename(file_tmp.rsplit('.',1)[0])

    work_image.set_colors_to_use(get_random_colors(colors))

    print " Clipped amount: {}".format(work_image.error_sum)

    best_achieved_error = 10000000
    last_change = 0
    for x in range(2000):
        #print x, work_image.colors
        work_image.error_sum = 0
        work_image.dither(quiet = True)
        least_used = -1
        least_frequency = 100000

        for i, col in enumerate(work_image.work_colors):
            if least_frequency > col.count:
                least_frequency = col.count
                least_used = i
            print " {:>2} {:<13} :{:>8}   {:>5.2f}%".format(i, col, col.count, 100 * float(col.count) / work_image.x / work_image.y)
        #print " [{:>3}] Achieved {} vs. needed:  {}".format(x, least_frequency, float(work_image.x) * work_image.y / 2 / colors)
        current_error = float(work_image.error_sum) / work_image.x / work_image.y
        print " [{:>3}] Actual error: {:.3f}% (best: {:.3f}, last change: {} ago)".format(x, current_error, best_achieved_error, x - last_change)
        if current_error < best_achieved_error:
            best_achieved_error = current_error
            work_image.save_new_image()
            if saveworkimages == True:
                work_image.save_new_image(work_folder = True, partial = partial)
                work_image.work_files_counter += 1

            work_image.colors_to_final()
            last_change = x

        else:
            work_image.final_colors_to_work()

        work_image.mutate()
        work_image.clean_errors()

        if last_change + idleiterations < x:
            print " * processing of {} done ....".format(file_tmp)
            break

