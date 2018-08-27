'''
------------------------------------------------------------------------------------------------------------------------------------------------------
Author:	Tanmay Sawaji                                                                                                                                |
If you have any doubts or suggestions, you can contact me at tanmaysawaji44@gmail.com                                                                |
This code can undergo a lot of improvements as the logic used for selection, crossover and the fitness function is very basic                        |
The neural network used for making predictions is structured randomly and the inputs taken can be improved                                           |
The reason for me to write this code was that I have always read about genetic algorithm in theory but I had never seen it implemented practically   |
I had a lot of doubts on the topic which were solved when I decided to tackle this problem                                                           |
This is a very rudimentary implementation of the algorithm and its only purpose is to educate anyone interested in the topic                         |
The code for snake game is not written by me, I have used the code from https://gist.github.com/sanchitgangwar/2158089                               |
------------------------------------------------------------------------------------------------------------------------------------------------------
'''
import random
import curses
from curses import KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN
from random import randint
import numpy as np 
import math
from matplotlib import pyplot as plt 

'''
The population size is fixed at 50
The top 5 snakes will be selected for crossover
The top 2 snakes will be used as they are in the next generation so that the performance does not diminish
'''

#Defines activation function for neurons
def activation_function_exp(z):
	res = 1.0 / (1+math.exp(-z))
	return res

#This function accepts the direction of the snake and sets the "key" parameter to that direction 
def get_key(index, snake_direction):
	if snake_direction == 'up':
		if index == 0:
			key = 260
		elif index == 1:
			key = 261
		else:
			key = 259
	elif snake_direction == 'down':
		if index == 0:
			key= 261
		elif index == 1:
			key = 260
		else: 
			key = 258
	elif snake_direction == 'left':
		if index == 0:
			key = 258
		elif index == 1:
			key = 259
		else:
			key = 260
	else:
		if index == 0:
			key = 259
		elif index == 1:
			key = 258
		else:
			key = 261
	return key

#This function initiates the population by randomizing all the values
def start_population():
	population = []
	genome = []
	for i in range(50):
		for j in range(54):
			gene = random.random()
			genome.append(gene)
		population.append(genome)
		genome = []
	return population

#This function finds out the direction in which the snake is moving
def find_direction_snake(snake):
	direction = ''
	head = snake[0]
	second = snake[1]
	if head[0] == second[0]:
		if head[1] < second[1]:
			direction = 'up'
		else:
			direction = 'down'
	else:
		if head[0] < second[0]:
			direction = 'left'
		else:
			direction = 'right'
	return direction

#This function finds out whether there are obstacles present to the left, right or ahead of the snake
#The values returned are boolean(not True or False but 1 or 0)
def find_obstacle_side(snake_direction, snake):
	obs_straight = 0
	obs_left = 0
	obs_right = 0
	if snake_direction == 'up':
		next_pos = [snake[0][0], snake[0][1]-1]
		left_pos = [snake[0][0]-1, snake[0][1]]
		right_pos = [snake[0][0]+1, snake[0][1]]
		if next_pos in snake or next_pos[1] == 0:
			obs_straight = 1
		if left_pos in snake or left_pos[0] == 0:
			obs_left = 1
		if right_pos in snake or right_pos[0] == 19:
			obs_right = 1
	elif snake_direction == 'down':
		next_pos = [snake[0][0], snake[0][1]+1]
		left_pos = [snake[0][0]+1, snake[0][1]]
		right_pos = [snake[0][0]-1, snake[0][1]]
		if next_pos in snake or next_pos[1] == 59:
			obs_straight = 1
		if left_pos in snake or left_pos == 19:
			obs_left = 1
		if right_pos in snake or right_pos == 0:
			obs_right = 1
	elif snake_direction == 'left':
		next_pos = [snake[0][0]-1, snake[0][1]]
		left_pos = [snake[0][0], snake[0][1]+1]
		right_pos = [snake[0][0], snake[0][1]-1]
		if next_pos in snake or next_pos[0] == 0:
			obs_straight = 1
		if left_pos in snake or left_pos[1] == 59:
			obs_left = 1
		if right_pos in snake or right_pos[1] == 0:
			obs_right = 1
	elif snake_direction == 'right':
		next_pos = [snake[0][0]+1, snake[0][1]]
		left_pos = [snake[0][0], snake[0][1]-1]
		right_pos = [snake[0][0], snake[0][1]+1]
		if next_pos in snake or next_pos[0] == 19:
			obs_straight = 1
		if left_pos in snake or left_pos[1] == 0:
			obs_left = 1
		if right_pos in snake or right_pos[1] == 59:
			obs_right = 1
	return obs_left, obs_right, obs_straight

#This function finds out whether there is food present to the left, right or ahead of the snake
#The values returned are boolean(not True or False but 1 or 0)
def find_food_side(snake_direction, head, food):
	food_straight = 1
	food_left = 0
	food_right = 0
	if snake_direction == 'up':
		if food[0] < head[0]:
			food_left = 1
			food_right = 0
			food_straight = 0
		elif food[0] > head[0]:
			food_left = 0
			food_right = 1
			food_straight = 0
		else:
			food_left = 0
			food_right = 0 
			food_straight = 1
	elif snake_direction == 'down':
		if food[0] < head[0]:
			food_left = 0
			food_right = 1
			food_straight = 0
		elif food[0] > head[0]:
			food_left = 1
			food_right = 0
			food_straight = 0
		else:
			food_left = 0
			food_right = 0
			food_straight = 1
	elif snake_direction == 'left':
		if food[1] < head[1]:
			food_left = 0
			food_right = 1
			food_straight = 0
		elif food[1] > head[1]:
			food_left = 1
			food_right = 0
			food_straight = 0
		else:
			food_left = 0
			food_right = 0
			food_straight = 1
	elif snake_direction == 'right':
		if food[1] < head[1]:
			food_left = 1
			food_right = 0
			food_straight = 0
		elif food[1] > head[1]:
			food_left = 0
			food_right = 1
			food_straight = 0
	return food_left, food_right, food_straight

#This function includes the neural network to predict which direction should the snake go to
def predict_key(obs_left, obs_right, obs_straight, food_left, food_right, food_straight, genome):
	u = np.array(genome[:30]).reshape(5,6)
	v = np.array(genome[30:45]).reshape(3,5)
	w = np.array(genome[45:54]).reshape(3,3)
	a = [obs_left, obs_right, obs_straight, food_left, food_right, food_straight]
	b = []
	c = []
	d = []

	#The input layer contains 6 nodes
	#The first hidden layer contains 5 nodes
	#The second hidden layer contains 3 nodes
	#The output layer contains 3 nodes

	for i in range(5):
		total = 0
		for j in range(6):
			total += (u[i][j]*a[j])
		fnz = activation_function_exp(total)
		b.append(fnz)

	for i in range(3):
		total = 0
		for j in range(5):
			total += (v[i][j]*b[j])
		fnz = activation_function_exp(total)
		c.append(fnz)

	for i in range(3):
		total = 0
		for j in range(3):
			total += (w[i][j]*c[j])
		fnz = activation_function_exp(total)
		d.append(fnz)

	res = d[0]
	index = 0
	if res < d[1]:
		index = 1
		res = d[1]
		if res < d[2]:
			index = 2
			res = d[2]
	elif res < d[2]:
		index = 2
		res = d[2]
	return index

# This function includes the actual game which each member of the population plays once
def play_game(population):
	scores = []
	best_score = -999
	for rows in range(len(population)):
		curses.initscr()
		win = curses.newwin(20, 60, 0, 0)
		win.keypad(1)
		curses.noecho()
		curses.curs_set(0)
		win.border(0)
		win.nodelay(1)

		key = KEY_RIGHT                                                    # Initializing values
		score = 0

		snake = [[4,10], [4,9], [4,8]]                                     # Initial snake co-ordinates
		food = [10,20]                                                     # First food co-ordinates

		win.addch(food[0], food[1], '*')                                   # Prints the food
		while True:                                                		   # While Esc key is not pressed
		    win.border(0)
		    win.addstr(0, 2, 'Score : ' + str(score) + ' ')                # Printing 'Score' and
		    win.addstr(0, 27, ' SNAKE ')                                   # 'SNAKE' strings
		    win.timeout(int(150 - (len(snake)/5 + len(snake)/10))%120)          # Increases the speed of Snake as its length increases
		    
		    prevKey = key                                                  # Previous key pressed
		    # event = win.getch()
		    step = 0
		    # key = key if event == -1 else event
		    snake_direction = find_direction_snake(snake)
		    obs_left, obs_right, obs_straight = find_obstacle_side(snake_direction, snake)
		    food_left, food_right, food_straight = find_food_side(snake_direction, snake[0], food)

		    index = predict_key(obs_left, obs_right, obs_straight, food_left, food_right, food_straight, population[rows])
		    key = get_key(index, snake_direction)
		            # If snake runs over itself
		    if (key == 260 and prevKey == 261) or (key == 261 and prevKey == 260) or (key == 258 and prevKey == 259) or (key == 259 and prevKey == 258):
		        key = prevKey
		    elif snake[0] in snake[1:]:
		        break
		    # Calculates the new coordinates of the head of the snake. NOTE: len(snake) increases.
		    # This is taken care of later at [1].
		    snake.insert(0, [snake[0][0] + (key == KEY_DOWN and 1) + (key == KEY_UP and -1), snake[0][1] + (key == KEY_LEFT and -1) + (key == KEY_RIGHT and 1)])
		    # If the snake touches a boundary line, it dies
		    if snake[0][0] == 0: break
		    if snake[0][1] == 0: break
		    if snake[0][0] == 19: break
		    if snake[0][1] == 59: break
		    
		    if snake[0] == food:                                            # When snake eats the food
		        food = []
		        score += 1
		        while food == []:
		            food = [randint(1, 18), randint(1, 58)]                 # Calculating next food's coordinates
		            if food in snake: food = []
		        win.addch(food[0], food[1], '*')
		    else:    
		        last = snake.pop()                                          # [1] If it does not eat the food, length decreases
		        win.addch(last[0], last[1], ' ')
		    win.addch(snake[0][0], snake[0][1], '#')
		    step += 1
		curses.endwin()
		if best_score < score:
			best_score = score
		fitness = score - (0.005*step)
		scores.append(fitness)
	return scores, best_score

#This function defines how the best players are selected
def selection(fitness, population):
	parents = []
	for i in range(5):
		temp = fitness.index(max(fitness))
		parents.append((population[temp], fitness[temp]))
		del population[temp]
		del fitness[temp]
	return parents
		
#This function defines how the next generation players are formed by mixing genes of the parents
def crossover(parents):
	next_gen = []
	next_gen.append(parents[0][0])
	next_gen.append(parents[1][0])
	for i in range(5):
		if i is not 0 or i is not 1:
			temp = []
			for j in range(54):
				index = random.randint(0,4)
				temp.append(parents[index][0][j])
			next_gen.append(temp)
	return next_gen

#This function creates random mutations so that the code does not get stuck in a local maxima
def mutation(next_gen):
	for i in range(len(next_gen)):
		counter = 0
		for j in range(len(next_gen[i])):
			if random.random() < 0.3:
				next_gen[i][j] = random.random()
				counter += 1
			if counter > 10:
				break
	return next_gen

total_gens = 100
generation = 1
population = start_population()
best_player = []
performance = []

while generation <= total_gens:
	fitness, best_score = play_game(population)
	parents = selection(fitness, population)
	if not best_player:
		best_player.append((generation, parents[0][0], parents[0][1]))
		f = open('best_player.txt', 'w+')
		f.write("The best player is\n{}".format(best_player))
	else:
		if best_player[0][2] < parents[0][1]:
			del best_player[0]
			best_player.append((generation, parents[0][0], parents[0][1]))
			f = open('best_player.txt', 'w+')
			f.write("The best player is\n{}".format(best_player))
	# f = open('gen_'+str(generation)+'.txt', 'w+')
	# f.write("The best player is\n{}".format(best_player))
	next_gen = crossover(parents)
	del population
	population = mutation(next_gen)
	performance.append(best_score)
	print("Best player score is {} \nBest player score of this gen is {}".format(best_player[0][2], parents[0][1]))
	print("End of generation ",str(generation))

	generation += 1


fig = plt.figure(figsize = (20,20))
ax = fig.add_subplot(111)
ax.plot(range(100), performance, label = "Score", markeredgewidth = 1 , linewidth = 2)
plt.xlabel("Generation")
plt.ylabel("Top score")
plt.legend()
plt.show()
