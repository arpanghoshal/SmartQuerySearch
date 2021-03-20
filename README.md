# SmartQuerySearch
## How to use
1. pip3 install -r requirements.txt
2. python3 AnswerModel.py
3. python3 Recommendation.py
4. python3 AnswerAPI.py

Now, in 0.0.0.0:5051 -> Topic Detection will work
In 0.0.0.0:5052 -> Query answer wil work


## Format
#### GET Requests 

- Topic Detection: 0.0.0.0:5051/topic?query=<your-query>
- Smart Query Search: 0.0.0.0:5052/answer/<your-website>?query=<your-query>

Example:-
- 0.0.0.0:5051/topic/when apple was found
- 0.0.0.0:5052/answer/wikipedia/?query=when apple was found
