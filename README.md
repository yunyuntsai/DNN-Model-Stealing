# DNN Models Extraction
This is the repo for [CloudLeak: Large-Scale Deep Learning Models Stealing Through Adversarial Examples](https://www.ndss-symposium.org/wp-content/uploads/2020/02/24178.pdf), Honggang Yu, Kaichen Yang, Teng Zhang, Yun-Yun Tsai, Tsung-Yi Ho, Yier Jin in Proceeding of Network and Distributed System Security Symposium (NDSS), 2020. Our code is implemented in Python 3.6 and Caffe. 

The following figure illustrates the transfer framework for our proposed model extraction method: <br/>
![Alt text](https://user-images.githubusercontent.com/20013955/99144591-07a8ea00-26a2-11eb-899b-96a6b97016d8.PNG)<br/>
<br/>
(a) generate unlabeled adversarial examples as synthetic dataset. <br/>
(b) query victim model using the generated synthetic dataset. <br/>
(c) label adversarial examples according to the output of the victim model. <br/> 
(d) train the local substitute model using the synthetic dataset. <br/>
(e) use the local substitute model for predictions. The local substitute model is expected to match the performance of the victim model. <br/>

For more detail, please refer to our [slides](https://www.ndss-symposium.org/wp-content/uploads/24178-slides.pdf), and [video](https://www.youtube.com/watch?v=tSUQl85Hprs&list=PLfUWWM-POgQuRxr8iZwL_6Dw-isT5CHUt&index=3&t=0s).
