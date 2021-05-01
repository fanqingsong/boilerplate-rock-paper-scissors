# This entrypoint file to be used in development. Start by reading README.md
from RPS_game import play, mrugesh, abbey, quincy, kris, human, random_player
from RPS import player
from unittest import main

print("-------- you vs quincy -------------")
play(player, quincy, 1000)

print("-------- you vs abbey -------------")
play(player, abbey, 1000, False, 30)

print("-------- you vs kris -------------")
play(player, kris, 1000)

print("-------- you vs mrugesh -------------")
play(player, mrugesh, 1000)

# Uncomment line below to play interactively against a bot:
# play(human, abbey, 20, verbose=True, verbose=True)
#
# Uncomment line below to play against a bot that plays randomly:
# play(human, random_player, 1000, verbose=True)



# Uncomment line below to run unit tests automatically
# main(module='test_module', exit=False)