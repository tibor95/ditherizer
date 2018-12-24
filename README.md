Ditherizer is simple python (2.7) script to convert normal images into ditherized ones with low nuber of colors.

It uses simpliefied genetic algorithm to find out most suitable colors, that would be used for
dithering the image with least possible errors.


The script can be run like:

python img_ditherer.py -f gradient.png -s -c 4 -t 3


Help is available with '-h' argument of course, and rigt now look like:

Image ditherer, looking for optimal unique colors so that errors are as
minimal as possible

optional arguments:
  -h, --help            show this help message and exit
  -c COLORS, --colors COLORS
                        unique colors for final image
  -t THREADS, --threads THREADS
                        threads count
  -p PERCENTAGE, --percentage PERCENTAGE
                        for resizing the image, default 100. Smaller images
                        are processed faster of course
  -i IDLEITERATIONS, --idleiterations IDLEITERATIONS
                        The script terminates if n iterations brought no
                        improvement. Default = 75. Actually it waits
                        iterations/threads iterations.
  -o OUTFILE, --outfile OUTFILE
                        does not work for now
  -d OUTDIR, --outdir OUTDIR
                        target directory for output files
  -s, --saveworkimages  save work images (after iteration that brough
                        improvement)
  -A, --partial         Leave strips of original image on the sides for
                        comparison
  -f INFILE [INFILE ...], --infile INFILE [INFILE ...]
                        one of more images that will be processed
  -v, --verbosity       increase output verbosity



WIKI is available here:

https://github.com/tibor95/ditherizer/wiki
