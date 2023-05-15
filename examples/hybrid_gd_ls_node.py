"""
The idea of hybrid ALS-GD-Adjoint for Neural-ODE
http://www1.ece.neu.edu/~erdogmus/publications/C041_ICANN2003_LinearLSlearning_Oscar.pdf
https://papers.nips.cc/paper_files/paper/2017/file/393c55aea738548df743a186d15f3bef-Paper.pdf
http://www1.ece.neu.edu/~erdogmus/publications/C034_ESANN2003_Accelerating_Oscar.pdf
https://www.jmlr.org/papers/volume7/castillo06a/castillo06a.pdf
"""

# TODO
#   1. read the two papers
#   - http://www1.ece.neu.edu/~erdogmus/publications/C041_ICANN2003_LinearLSlearning_Oscar.pdf
#   - https://www.jmlr.org/papers/volume7/castillo06a/castillo06a.pdf
#   2- quick idea of paper https://papers.nips.cc/paper_files/paper/2017/file/393c55aea738548df743a186d15f3bef-Paper.pdf
#   3 -Finalize the EM-code for ode and documentation
#       3.1 - finalize the update method
#       3.2 - Apply Dopri step
#       https://numerary.readthedocs.io/en/latest/dormand-prince-method.html
#       https://core.ac.uk/download/pdf/237206461.pdf
#   4 - visualize the loss landscape
#   https://medium.com/mlearning-ai/visualising-the-loss-landscape-3a7bfa1c6fdf
#   5 - ALS-only neural ODE coding starting
#   6- Hybridize ALS-Adjoint based on Last-Layer- LS idea
#   7- make last step as RK45 or dopri5


