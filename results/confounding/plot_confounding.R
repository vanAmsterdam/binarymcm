# plot results of confounding experiment
library(data.table)
library(ggplot2); theme_set(theme_bw())
library(stringr)
library(purrr)
library(magrittr)

resdir <- here::here("results", "confounding")
figdir <- resdir
# figdir <- "~/git/fixedtx/writing/manuscript/figs/"

color_map = c(
  `fully observational` = 'red',
  'offset' = 'orange',
  'CR-MCM' = 'darkgreen',
  'ATE' = 'black'
)

rdf <- fread(file.path(resdir, 'settingdf.csv'))
rdfm <- melt(rdf, measure.vars = c('offset', 'constrained', 'ate', 'full'),
             value.name = 'pehe2',
             variable.name = 'estimator')
rdfm[, pehe:=sqrt(pehe2)]

rdfm[estimator=='ate', estimator:='ATE']
rdfm[estimator=='constrained', estimator:='CR-MCM']
rdfm[estimator=='full', estimator:='fully observational']
rdfm[, estimator:=factor(estimator)]
rdfm <- rdfm[gamma_ut_sign==1.]

rdfm[, `:=`(
  oru=round(exp(au), 2)
  )]
oru_levels = c(1,2,5)
oru_labels = c(
 expression(alpha[u]~'='~log(1)),
 expression(alpha[u]~'='~log(2)),
 expression(alpha[u]~'='~log(5))
)

rdfm[, oru_label:=factor(oru,
                         levels=oru_levels,
                         labels=oru_labels)]


ggplot(rdfm, aes(x=ax, y=pehe, col=estimator)) + 
  geom_point() + geom_line() + 
  facet_grid(~oru_label, labeller=label_parsed) +
  scale_color_manual(values=color_map) + 
  labs(y='PEHE', x=expression(alpha[x]))

width=20
ratio=2.5
height=width / ratio


ggsave(file.path(figdir, 'confounding.pdf'),
       width=width, height=height, units='cm', dpi=300)
