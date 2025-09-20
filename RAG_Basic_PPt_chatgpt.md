RAG





Prerequistic: Basic Python , Basic RestApi



Editor : Google Collab , VS code



packages : Langchain , Llamaindex







why and what RAG for chatbot ?



RAG = Retrieval Augmented Generation  



&nbsp;LLM - Large Language Model



&nbsp;Tokens - 3.8 Character 



&nbsp;Context size - (size of LLM taken input Text)

&nbsp;

&nbsp;Knowledge cutoff 









&nbsp;                                     RAG Stages 01





1\) Documents Loaders ( pdf, links  etc)



2\) Documents Chunking ( we split or break down the large size file and store it in Vector Database)



&nbsp;     Vector Base - it will convert the characters into machine code to store ( mile verse )





&nbsp;                                    RAG Stages 02







1\) Retrieval ( it will take and search it in vector store to find the question asked by users )



2\) Prompt engineering ( by users)



3\) Answer Generation ( in this LLm will change the machine code to human language give us answer)







&nbsp;                                     Hands on  Implementation 



&nbsp;                                           Stage 01 ( pdf, links , import )



Code in python : 

&nbsp; 

&nbsp;		#import pdf, links Loaders  

&nbsp;		!pip install langchain-community --quiet 

&nbsp;		!pip install pypdf --quiet





&nbsp;		#connection for Lang chain 



&nbsp;		from langchain.document.loaders import pyPDFLoader 



&nbsp;		#Loders 

&nbsp;

&nbsp;		loader = pyPDFLoader("")

&nbsp;		pdf\_pages=loader.load()

&nbsp;		

&nbsp;		pdf\_pages\[0]





&nbsp;		# youtube video loading  



&nbsp;		Youtube video --->> Audio fetch ---->> Transcription Generation ---->> Text File .



&nbsp;		#Youtube Loader



&nbsp;		!pip install yt\_dip --quiet 

&nbsp;		!pip install pydub -- quiet 

&nbsp;		!pip install faster-whisper --quiet 



&nbsp;		# Lang chain for youtube



&nbsp;		from langchain.document\_loaders.generic import GenericLoader 

&nbsp;		from langchain.\_community.document\_loaders.parsers.audio import FasterwhisperParser 

&nbsp;		from langchain.document\_loaders.blob\_loaders.youtube\_audio import YoutubeAudioLoader





&nbsp;		#go to GitHub and take code from faster Whisper 

&nbsp;

&nbsp;		add line of code..........

&nbsp;		..........................

&nbsp;		..........................

&nbsp;		..........................



&nbsp;		#to view

&nbsp;		docs\[1]



&nbsp;		#to length

&nbsp;		len(docs)



&nbsp;		#to combine two doc



&nbsp;		combined\_docs= pdf\_pages + docs

&nbsp;		len(combined\_docs)





&nbsp;					Chunking  in vector Data Base (link : chunkviz.up.railway.app)





in vector they used splitter for chunking , splitter, chunk size, chunk overlap .





code for chunking :



&nbsp;		from langchain.text\_splitter import RecursiveCharacterTextSplitter



&nbsp;		chunk\_size=1043

&nbsp;		chunk\_overlap = 200 





&nbsp;		splitter = recursivecharacterTextSplitter(

&nbsp;				 chunk\_size=chunk\_size,

&nbsp;				chunk\_overlap=chunk\_overlap

&nbsp;				)



&nbsp;		chunked\_docs=splitter.split\_documents(combined\_docs)

&nbsp;		len(chunked\_docs)







&nbsp;					Vector Store \_Embedding 





&nbsp;  For example to convert charater or string to numbers to store in vector database



&nbsp;  we use embedding for semantic search in documents 





&nbsp;		input ---> embedding ---> numbers 



&nbsp;		input ----> LLm ---> text format 





open sources hub : Hugging Face ( we goona use from hugging face )



&nbsp;				go to ---->> hugging Face ---> Search for embedding Leaderboard --> find model



&nbsp;				code:



&nbsp;				#model loading  for embedding 

&nbsp;				 

&nbsp;				from langchain\_community.embeddings import HuggingFaceEmbeddings 

&nbsp;				all\_minlm\_embedding = HuggingFaceEmbedding( model\_name = " all-MinLM-l6-v2")

&nbsp;				multilingual\_embeddings = HuggingFaceEmbeddings(model\_name='intfloat/multilingual-e5-large')





&nbsp;				#to find the similarity 



&nbsp;				import numpy as np 

&nbsp;				np.dot(e1,e2)

&nbsp;	





&nbsp;					why Vector Database ?



&nbsp;				it will use similarity check 





&nbsp;			\*\*\*\*\*\*\*\*\*Photo of vector base \*\*\*\*\*\*\*\*\*



&nbsp;		we goona use Chroma ,HNSW , Distance function , like this it will group the  word semantic and similarity .









&nbsp;					Vector DB --Chroma 





&nbsp;	code : 



&nbsp;			#coding 



&nbsp;			!pip install chromadb --quiet 

&nbsp;			from langchain.vectorstores import Chroma



&nbsp;			persist\_directory = "/db/chroma/"

&nbsp;			

&nbsp;			#code for vector base



&nbsp;			vectordb = Chroma.from\_documents(



&nbsp;					documents=chunked\_docs

&nbsp;					embedding= multilingual\_embedding 

&nbsp;					persist\_directory = persist\_directory



&nbsp;					)





&nbsp;			#searching and working vEctorDb



&nbsp;			queston = "how are joseph "

&nbsp;			vectordb.similarity\_search(question,k=3)













&nbsp;					Retrieval 





&nbsp;		code:



&nbsp;			%% max\_marginal\_search\_ will return match words from database



&nbsp;		



&nbsp;			#Meta Data filtering 



&nbsp;			

&nbsp;			vector.dbsimilarity\_search (

&nbsp;				question,

&nbsp;				k=3,

&nbsp;				filters={"source: ""/content/embedded-systems.pdf"}







&nbsp;					Prompt engineering 



&nbsp;# for ai Keys ...........

&nbsp;				go to ---> Groq ----> click Grop\_Api ---> api\_key







&nbsp;					LLM Parameter 







&nbsp;		configure the LLM :

&nbsp;

&nbsp;				Temperature === Creativity rate



&nbsp;				Max Completion Tokens === size of the answer paragraph 



&nbsp;				Top P === Almost same for Temperature 



&nbsp;				Seed === will use to share to friends to this model with same model parameter we 					generated

&nbsp;					

&nbsp;				Stop Sequence ==== use stop loke Delimiter in power Bi







&nbsp;				

&nbsp;                                    LLM (Llama) 





&nbsp;			The                    Tokens || Logic 

&nbsp;						forest || 8.2





&nbsp;					new\_logic = logit /temperature 





&nbsp;					Top P === Cumulative sum of give numbers ..... (vice versa)   







&nbsp;					



&nbsp;					Grop API CODING 











&nbsp;	Code : 





&nbsp;		!pip install langchain-grop --quiet 



&nbsp;		# add grop api key 



&nbsp;		import getpass 

&nbsp;		import os 

&nbsp;		

&nbsp;		if "GROP\_API\_KEY " not in os.environ :

&nbsp;			os.environ\["GROP\_API\_KEY "] = getpass.getpass("Enter your Grop API KEY : " )







&nbsp;		# addd codding gropapi 



&nbsp;		

&nbsp;	from langchain\_grop import chatGrop 



&nbsp;	llm=chatGrop(

&nbsp;		model="llam-3.1-8b-instant",

&nbsp;		temperature =0 ,

&nbsp;		max\_tokens = 250 , 



&nbsp;		)



&nbsp;	llm.invoke("write a poem about nature " )



&nbsp;	answer....................

&nbsp;	..........................

&nbsp;	.........................

&nbsp;	......................









&nbsp;					Prompt Engineering 





&nbsp;			we need assign like this 



&nbsp;				1) Role ( finance , Healthcare )   ............. system prompt 



&nbsp;				2) Instruction ( use case you are analyst ) .......  system Prompt 



&nbsp;				3) context -- adding pdf 



&nbsp;				4) Examples --- zero shot , single shot 

&nbsp;		

&nbsp;				

&nbsp;			Follow of model 

&nbsp;			 



&nbsp;			System Prompt 



&nbsp;		messages = \[ { "role" : " System " , "content " : "hi you are historian " },

&nbsp;				 { "role" : " user" , "content " : "where is rome " }, ]

&nbsp;				

&nbsp;			tool coning ............

&nbsp;	

&nbsp;			llm Generation 



&nbsp;			-----To get Good Prompt got LangchainHub search for prompt





&nbsp;			



&nbsp;					Flow chart Of our project 





&nbsp;								     ------>> system promt





&nbsp;			user(Question) ------>>> Vector DB (chroma ) ------>> Chunks //context ---->> 

&nbsp;								

&nbsp;								     ------->> Questions 



&nbsp;			-->>LLM (grop-Llama) ---->> Answer Generation 

&nbsp; 					



&nbsp;			



&nbsp;		





&nbsp;		







&nbsp;					Prompt Engineering 





\#code 





&nbsp;			system\_promt = { " add prompt " } 



&nbsp;			from lamgchain ...............



&nbsp;			system message ................



&nbsp;			# ask question 



&nbsp;			question = " ..........."



&nbsp;			#vector DB Calling 



&nbsp;			docs= vectordb.similarity.......





&nbsp;			# use pandas 



&nbsp;                        code....................

&nbsp;			

&nbsp;				context = join all the string 



&nbsp;			# add human Message 

&nbsp;			

&nbsp;				human\_response =\[Humanmessage(content = context + question ) ]

&nbsp;				reponse= llm.invoke(system\_message + human\_message)



&nbsp;			



&nbsp;				

&nbsp;		 

&nbsp;						Memory Saving the conversation 





&nbsp;		in Lang graph we can save conversation 





&nbsp;		code : 





&nbsp;		!pip install langgraph --quiet 



&nbsp;		code from langchain.............







&nbsp;		the langgrap 



&nbsp;		app.invoke ( {"messages" : \[HumanMessage(content ="what did i asked " )]},



&nbsp;				config = { "configurable " : { "thread\_id" : "1')}}, )







&nbsp;		dont change the Thread id , if we change the thread id the conversation is forgot by the llm .













&nbsp;		use  Stremlit to add ui to project 









&nbsp;					**Final Application ..........** 





&nbsp;



&nbsp;			

















&nbsp;			



























&nbsp;			

&nbsp;			

&nbsp;









&nbsp;		



&nbsp;		 

&nbsp;		

&nbsp;	

&nbsp;		

&nbsp; 







&nbsp;		











&nbsp;		 





&nbsp;	



&nbsp;		

&nbsp;		

&nbsp;

&nbsp;		



&nbsp;		

&nbsp;		









&nbsp;                                   





&nbsp;  

