import csv
from collections import defaultdict

user_ratings = defaultdict(list)
movie_ratings = defaultdict(list)

with open("csv/arb_test.csv", "rb") as f:
    reader = csv.reader(f, delimiter=",")
    for index, line in enumerate(reader):
        user = line[0]
        movie = line[1]
        rating = line[2]

        user_ratings[user].append(rating)
        movie_ratings[movie].append(rating)

print "User rating counts 1711: " + str(len(user_ratings['1711']))

movie1 = movie_ratings['254']
sum = 0
for rating in movie1:
    sum = sum + int(rating)

movie1_average = float(sum) / len(movie1)


print "Sum: " + str(sum)
print "Number of ratings: " + str(len(movie1))
print "Average rating: " +  str(movie1_average)
