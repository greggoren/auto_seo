from Multi_document_summary.diversification import diversify

sentences = ["A toilet can be designed for people who prefer to sit (by using a toilet pedestal) or for people who prefer to squat and use a squat toilet.","A toilet is designed for people who prefer to sit (by using a toilet pedestal) or for people who prefer to use a squat toilet."]
scores={1:1.0,2:0.3,3:0.4,4:1.0}
transition = {1:{2:0.2,3:0.3,4:0.96},2:{1:0.3,3:0.3,4:0.2},3:{1:0.3,2:0.3,4:0.2},4:{1:0.95,3:0.3,2:0.2}}
diversify(scores,transition,2,{1:4,2:5,3:6},{1:{1:2,2:4,3:7},2:{1:1,2:2,4:1},3:{1:5,2:20,3:7},4:{1:4,2:5,3:6.5}},{1:3,2:4,3:5.5},0.5)
