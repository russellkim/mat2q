             +----------------+
             |  PlannerAgent  |
             +-------+--------+
                     |
           High-level SQL Plan
                     |
        +------------+-------------+
        |            |             |
 +-------------+ +-------------+ +-------------+
 | SQLAgent #1 | | SQLAgent #2 | | SQLAgent #3 |
 +------+------+ +------+------+ +------+------+
        |                   |                   |
    SQL Candidate 1    SQL Candidate 2     SQL Candidate 3
        \                   |                   /
         \                  |                  /
          \       +------------------+       /
           \----> |  Verifier Agent  | <----/
                 +--------+---------+
                          |
                  Best Executable SQL

