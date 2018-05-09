# eDreams ODIGEO Baggage Likelihood prediction model - personal log

## Day 1

First thoughts looking at the data



Preliminary 80/20 train/validation split

I would imagine that flight distance would account for a lot of the probility of  of luggage selection (high R2), as people who travel further I would assume need to carry more than if they are doing a short weekend trip withing Europe.

I think the best would be to try a good old mulitple logisitic regression model with several dummy variables, but omit data which at first you find useless.

Data which seems irrelevant at first:
TIMESTAMP
DEPARTURE
ARRIVAL



Maybe those who pick SMS as an extra are more likely to pick extras ? To investigate.

Adults travelling alone I would assume would be less likely to book luggage, but with one or more children much less likely to get luggage

Interestingly flight DISTANCE seems to follow a bimodal distribution with alot of short flights between 1500-2500km and alot between 5000-6000km.


