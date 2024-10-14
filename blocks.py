# Looking at the the tasks from the Abstraction and Reasoning Corpus, while each task is unique, it is clear that there are certain concepts,
# such as operations like rotation or mirroring, that occur repeatedly throughout the corpus. What seems feasible is to think of a set of
# building blocks that encapsulate those concepts that can then be used to build solution programs, that is, task-specific programs that correctly
# transform each of the input grids of a given task into its corresponding output grid. Such a set of building blocks is a form of domain-specific
# language (DSL). A DSL defines a set of programs that it can express, and the process of finding or creating such a program solving a given task
# is a form of program synthesis. Building a good DSL that well captures the explicitly stated core knowledge priors of ARC in an abstract and
# combinable manner, combined with an adequate program synthesis approach is suggested as a possible way to tackling ARC by its creator François
# in On the Measure of Intelligence.

# This notebook presents a very simple such DSL tailored to ARC and how it can be used to perform program synthesis, intended to serve as a starting
# point for anyone new to the ARC benchmark. For the sake of demonstration, the DSL as well as the program synthesis are overly simplistic here,
# to the point of being naive, as will be demonstrated. First, the DSL is defined as some basic functions, also called primitives, that transform
# grids. Second, program synthesis as search over compositions of those primitives is performed. The search is naive in that it is a brute force
# search over all possible primitive compositions up to a certain depth that is done for each task and simply uses the first program it finds to
# solve the training examples to make predictions for the test examples.

# It is obvious why some tasks can't be solved by the above DSL, but one simple proof would be the following: No primitive ever increases the pixel
# count of a grid, hence neither can any composition of the primitives ever do so - and since for certain tasks, the output grids do have more pixels
# than the input grids, the DSL is incomplete. To increase the expressibity of the program space (disregarding the maximum program size), one will
# want to expand the set of the primitives and also extend the structure of the considered programs beyond mere composition. Maybe it is a good idea
# to have primitives which take more than one input argument, or primitives that operate on types other than only grids, such as objects or integers.
# Note that viewing the transformations from inputs to outputs as a linear function composition is very misleading, as many tasks can't be neatly
# squeezed into this form: Some tasks seem much better addressed on a pixel- or object-level than on a grid-level. A good DSL is probably concise and
# allows expressing solutions to many tasks as short programs. Such a DSL may best be built by bootstrapping, that is, building a minimal version of
# it and then iterating back and forth between using it to solve ARC tasks and expanding it to account for unsolvable ARC tasks, all while having
# abstractness and flexibility of the primitives and how they can interplay in mind.

# Geometry
#     do nothing
#     rotate
#     mirror
#     shift image
#     crop image background
#     draw border
# Objects
#     rotate
#     mirror
#     shirt objects
#     move two objects together
#     move objects to edge
#     extend
#     repeat an object
#     delete an object
#     count unique objects and select the object that appears the most times
#     create pattern based on imxage colors
#     overlay object
#     replace objects
# Coloring
#     select colors for objects
#     select dominant/smallest color in image
#     denoise
#     fill in empty spaces
# Lines
#     color edges
#     extrapolate a straight/diagonal line
#     draw a line between two dots / or inersections between such lines
#     draw a spiral
# Grids
#     select grid squares with most pixels
#     scale to amke the inout and output the same size?
# Patterns
#     complete a symmetrical/repeating pattern
# Subtasks
#     object detection
#     object cohesion
#     object seperation
#     object persistance
#     counting or sorting objects
