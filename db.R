library(caret)
library(ggplot2)
library(dplyr)
library(gridExtra)
library(reshape)

#data generation
N <- 200 # number of points per class
D <- 2 # dimensionality
K <- 4 # number of classes
X <- data.frame() # data matrix (each row = single example)
y <- data.frame() # class labels

set.seed(118)

for (j in (1:K)){
  r <- seq(0.05,1,length.out = N) # radius
  t <- seq((j-1)*4.7,j*4.7, length.out = N) + rnorm(N, sd = 0.3) # theta
  Xtemp <- data.frame(x =r*sin(t) , y = r*cos(t)) #np.c_[r*np.sin(t), r*np.cos(t)]
  ytemp <- data.frame(matrix(j, N, 1))
  X <- rbind(X, Xtemp)
  y <- rbind(y, ytemp)
}

data <- data.frame(X, lable = y)
colnames(data) <- c(colnames(X), 'label')
data$label <- factor(data$label)

#creat grid for decision boundary
x_min <- min(X[,1])-0.1
x_max <- max(X[,1])+0.1

y_min <- min(X[,2])-0.1
y_max <- max(X[,2])+0.1

hs <- 0.01
grid <- data.frame(expand.grid(seq(x_min, x_max, by = hs), seq(y_min, y_max, by = hs)))
colnames(grid) <- c('x', 'y')

#train using different models
nnet.fit <- train(label ~ ., data=data, method="nnet", tuneGrid=data.frame(size = 50, decay = 0.01), 
                  trControl=trainControl(method="cv", number=3))

svm.fit.1 <- train(label ~ ., data=data, method="svmLinear", 
                  trControl=trainControl(method="cv", number=3))

svm.fit.2 <- train(label ~ ., data=data, method="svmRadial", 
                   trControl=trainControl(method="cv", number=3))

rf.fit <- train(label ~ ., data=data, method="rf", 
                  trControl=trainControl(method="cv", number=3))

models.name <- c('SVM with linear kernal', 'SVM with RBF kernal', 'Neural network', 'Random forest') 
models <- list(svm.fit.1, svm.fit.2, nnet.fit, rf.fit)

#combine predictions on grid
pred <- grid

plot.list <- NULL
for (i in 1:4){
  print(i)
  Z <- predict(models[[i]], grid)
  pred <- cbind(pred, Z)

}

colnames(pred) <- c('x', 'y', models.name) 
#convert from wide to long, prepare data for facet in ggplot
pred.melt <- melt(pred, id = c('x', 'y'))
colnames(pred.melt) <- c('x', 'y', 'model', 'label') 

#generate decision boundaries
dbPlot <- ggplot(data = pred.melt)+
  geom_tile(aes(x = x, y = y,fill = as.character(label)), alpha = 0.3, show.legend = F)+ 
  geom_point(data = data, aes(x=x, y=y, color = as.character(label)), size = 2) + 
  theme_bw(base_size = 20) + facet_wrap(~model) +
  ggtitle('Decision Boundaries')+
  theme(axis.ticks=element_blank(), axis.title=element_blank(),
        axis.text=element_blank(),legend.position = 'none',
        panel.grid.major = element_blank(), panel.grid.minor = element_blank())

#output png
png('dbPlot.png', width=1280, height=800)
dbPlot
dev.off()

