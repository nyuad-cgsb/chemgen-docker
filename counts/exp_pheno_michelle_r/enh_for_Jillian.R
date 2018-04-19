#load and format data
enhancer<-read.csv("enh_query.csv")
enhancerdf<- as.data.frame(enhancer)
library(plyr)
newdata<-arrange(enhancerdf,desc(date))

#more formatting and add calculated columns for 
enhancerdf$count_larva<-as.numeric(as.character(enhancerdf$count_larva))
enhancerdf$count_embryo<-as.numeric(as.character(enhancerdf$count_embryo))
enhancerdf$count_adult<-as.numeric(as.character(enhancerdf$count_adult))
enhancerdf$larvaperadult<- enhancerdf$count_larva / enhancerdf$count_adult
enhancerdf$broodsize<- enhancerdf$count_embryo + enhancerdf$count_larva
enhancerdf$survival<- enhancerdf$count_larva / (enhancerdf$count_embryo + enhancerdf$count_larva)

#check class of date column
class(enhancerdf$date)
#change class of date column to type Date
enhancerdf$date<-as.Date(as.character(enhancerdf$date))

#subset df to include only experiments done after Feb 1, 2015 (enhancer set)
enhancerdf <- subset(enhancerdf, date > as.Date("2015-02-01"))

highsurvival <- subset(enhancerdf, survival>= .7)

lowsurvival <- subset(enhancerdf, survival< .7)

library(ggplot2)

enhden <- ggplot(lowsurvival, aes(x=survival)) + 
  geom_density() +
  xlab("Devstar Survival")+
  facet_wrap(~ gene)
enhden

enhden2 <- ggplot(enhancerdf, aes(x=survival)) + 
  geom_density() +
  xlab("Devstar Survival")+
  facet_wrap(~ gene)
enhden2

##for N2s
enhancer_N2<-read.csv("enh_N2.csv")
enhancerdf_N2<- as.data.frame(enhancer_N2)
library(plyr)

enhancerdf_N2$count_larva<-as.numeric(as.character(enhancerdf_N2$count_larva))
enhancerdf_N2$count_embryo<-as.numeric(as.character(enhancerdf_N2$count_embryo))
enhancerdf_N2$count_adult<-as.numeric(as.character(enhancerdf_N2$count_adult))
enhancerdf_N2$larvaperadult<- enhancerdf_N2$count_larva / enhancerdf_N2$count_adult
enhancerdf_N2$broodsize<- enhancerdf_N2$count_embryo + enhancerdf_N2$count_larva
enhancerdf_N2$survival<- enhancerdf_N2$count_larva / (enhancerdf_N2$count_embryo + enhancerdf_N2$count_larva)


enhancerdf_N2$date<-as.Date(as.character(enhancerdf_N2$date))

enhancerdf_N2 <- subset(enhancerdf_N2, date > as.Date("2015-02-01"))

##take mean of N2x RNAi
enh_N2_mean<-ddply(enhancerdf_N2, .(gene, date, library_stock_id), summarize, mean=mean(survival))

N2com <- ggplot(enh_N2_mean, aes(x=mean)) + 
  geom_density() +
  xlab("Devstar Survival") +
  facet_wrap(~ date)
N2com

##using non- quantile normalized N2 (discuss with Kris)

enh_merge2 <- merge(enhancerdf, enh_N2_mean, by = c("date", "library_stock_id"))

enh_comparison_full <- subset(enh_merge2, survival <= .4 & mean >=.2)

enh_comparison_full2 <- subset(enh_comparison_full, select = c(gene, date, library_stock_id, 
                                                               experiment_id, survival, mean))
#density plot, all things with N2 mean survival>.2 and experimental survival < .4
enhcom3 <- ggplot(enh_comparison_full2x, aes(x=survival)) + 
  geom_density() +
  xlab("Devstar Survival")+
  facet_wrap(~ gene)
enhcom3

enh_comparison_full2$diff <- (enh_comparison_full2$mean - enh_comparison_full2$survival)

enh_comparison_full2$color <- ifelse(enh_comparison_full2$diff < 0.5,0,1)

#color where diff N2 survival and exp survival is >.5
enhcom4 <- ggplot(enh_comparison_full2, aes(x=survival, y=mean, group=1, color= color)) + 
  geom_point() +
  xlab("RNAi Survival")+
  ylab("N2 Survival")+
  facet_wrap(~ gene)
enhcom4

enh_comparison2 <- subset(enh_comparison2, date > as.Date("2015-02-01"))

###
#####
enh_merge2$diff <- (enh_merge2$mean - enh_merge2$survival)

enh_merge2$color <- ifelse(enh_merge2$diff < 0.5,0,1)

#color where diff N2 survival and exp survival is >.5
enhcom6 <- ggplot(enh_merge2, aes(x=survival, y=mean, group=1, color= color)) + 
  geom_point() +
  xlab("RNAi Survival")+
  ylab("N2 Survival")+
  facet_wrap(~ gene.x)
enhcom6

###
#count positives where RNAi survival <= .4 & N2 mean survival >=.2
enh_pos<-ddply(enh_comparison_full2, .(gene, date, library_stock_id), 
               summarize, count=length(library_stock_id))

enh_pos2<-ddply(enh_pos, .(gene), 
                summarize, count=length(gene))

#count difference is > .5

##to change cutoff: enh_comparison_full2$color <- ifelse(enh_comparison_full2$diff < 0.5,0,1)

enh_pos_diffx <- subset(enh_comparison_full2x, color == 1)

enh_diffx<-ddply(enh_pos_diffx, .(gene, date, library_stock_id), 
                 summarize, count=length(library_stock_id))

enh_diff1x<-ddply(enh_pos_diffx, .(gene), 
                  summarize, count=length(gene))

enh_diff2x<-ddply(enh_diffx, .(gene), 
                  summarize, count=length(gene))

###
#look for ste phenotype, full
enh_ste_check <- subset(enh_merge2, gene.x == "dhc-1")
average<- mean(enh_ste_check$broodsize, na.rm=TRUE)

enh_ste_check<-ddply(enh_merge2, .(gene.x, date), 
                     summarize, avg_broodsize=mean(broodsize, na.rm=TRUE), 
                     med_broodsize=median(broodsize, na.rm=TRUE), count= length(gene.x))

enh_ste <- subset(enh_merge2, broodsize <200)

enh_ste_count<-ddply(enh_ste, .(gene.x, date, library_stock_id), 
                     summarize, count=length(library_stock_id))

enh_ste_count2<-ddply(enh_ste_count, .(gene.x), 
                      summarize, count=length(gene.x))

#look for LA phenotype , emb-8
enh_LA <- subset(enh_merge, count_adult <4)

enh_LA2 <- subset(enh_LA, select = c(gene.x, gene.y, date, library_stock_id, 
                                     experiment_id, survival, mean))
enh_LA2<- subset(enh_LA2, date > as.Date("2015-02-01"))


#load and format data
enhL4440<-read.csv("enh_L4440.csv")
enhL4440df<- as.data.frame(enhL4440)
library(plyr)
newdata<-arrange(enhL4440df,desc(date))

#more formatting and add calculated columns for 
enhL4440df$count_larva<-as.numeric(as.character(enhL4440df$count_larva))
enhL4440df$count_embryo<-as.numeric(as.character(enhL4440df$count_embryo))
enhL4440df$count_adult<-as.numeric(as.character(enhL4440df$count_adult))
enhL4440df$larvaperadult<- enhL4440df$count_larva / enhL4440df$count_adult
enhL4440df$broodsize<- enhL4440df$count_embryo + enhL4440df$count_larva
enhL4440df$survival<- enhL4440df$count_larva / (enhL4440df$count_embryo + enhL4440df$count_larva)

#change class of date column to type Date
enhL4440df$date<-as.Date(as.character(enhL4440df$date))

#subset df to include only experiments done after Feb 1, 2015 (enhL4440 set)
enhL4440df <- subset(enhL4440df, date > as.Date("2015-02-01"))

enhL4440df<- separate(enhL4440df, experiment_id, c("plate", "well"), "_")


enhL4440den2 <- ggplot(enhL4440df, aes(x=survival, color=as.character(date), group=plate)) + 
  geom_density() +
  xlab("Devstar Survival")+
  facet_wrap(~ gene)
enhL4440den2

enh_L4440_dist<-ddply(enhL4440df, .(gene, date), 
                      summarize, mean=mean(survival))

enhL4440df_sum<-ddply(enhL4440df, .(gene, date), 
                      summarize, mean=mean(survival), sd= sd(survival))

enh_L4440_merge <- merge(enhancerdf, enhL4440df_sum, by = c("date", "gene"))

enh_L4440_merge$z <- (enh_L4440_merge$survival - 
                        enh_L4440_merge$mean)/enh_L4440_merge$sd

enh_L4440_merge$norm <- pnorm(enh_L4440_merge$z)

enh_L4440_merge$norm2 <- 1- enh_L4440_merge$norm

enhL4440norm <- ggplot(enh_L4440_merge, aes(x=z, color=as.character(date))) + 
  geom_density() +
  #xlab("Devstar Survival")+
  facet_wrap(~ gene)
enhL4440norm

enhL4440N2 <- ggplot(enh_L4440_N2_pos, aes(x=survival, color=as.character(date))) + 
  geom_density() +
  #xlab("Devstar Survival")+
  facet_wrap(~ gene.x)
enhL4440N2

enh_L4440_pos2 <- subset(enh_L4440_merge,enh_L4440_merge$norm2 > .98 )

enh_diff_l4440<-ddply(enh_L4440_pos2, .(gene, date, library_stock_id), 
                      summarize, count=length(library_stock_id))

enh_diff1x_L4440<-ddply(enh_diff_l44403, .(gene), 
                        summarize, count=length(gene))

enh_L4440_N2_merge <- merge(enh_L4440_merge, enh_N2_mean, by = c("date", "library_stock_id"))

enh_L4440_N2_pos <- subset(enh_L4440_N2_merge, norm2 > .98 & mean.y > .5 )

enh_diff_l4440_N2<-ddply(enh_L4440_N2_pos, .(gene.x, date, library_stock_id), 
                         summarize, count=length(library_stock_id))

enh_diff_l4440_N2_2 <- subset(enh_diff_l4440_N2, date >as.Date("2015-02-01"))
enh_diff_l44403 <- subset(enh_diff_l4440, count > 2 )

enh_diff_l4440_N23 <- subset(enh_diff_l4440_N2_2, count > 2 )


enh_diff1x_L4440_N2<-ddply(enh_diff_l4440_N23, .(gene.x), 
                           summarize, count=length(gene.x))

test <- enh_L4440_merge 

library(dplyr)
library(tidyr)

test2 <- test %>% group_by(gene) %>% mutate(nn = 1- pnorm(z)) %>% ungroup()


ptest <- ggplot(test, aes(x=nn, color=as.character(date))) + 
  geom_density() +
  #xlab("Devstar Survival")+
  facet_wrap(~ gene)
ptest

test2 <- data.frame(x=runif(10),let=rep(letters[1:5],each=2))
test2 <- test2 %>% group_by(let) %>% mutate(nn = sum(x)) %>% ungroup()






