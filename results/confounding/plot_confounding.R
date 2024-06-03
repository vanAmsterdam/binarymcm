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

id_vars <- c("settingidx", "ax", "au", "atu", "gamma_ut_sign", "a0", "at", 
  "atx", "axu", "atxu", "pt", "pu", "px")
pehe_vars <- c('offset', 'constrained', 'ate', 'full')

rdfm <- melt(rdf, id.vars=id_vars)
rdfm[variable%in%pehe_vars, variable:=paste0("pehe2_", variable)]
rdfm[, estimator:=str_extract(variable, "(?<=_)(.*)")]
rdfm[, metric:=str_replace(variable, "_(.*)", "")]
rdfm_pehe2 <- copy(rdfm[metric=="pehe2"])
rdfm_pehe2[, `:=`(metric="pehe", value=sqrt(value))]
rdfm <- rbindlist(list(rdfm, rdfm_pehe2), use.names=T)
rdfm[, variable:=NULL]

rdfm[estimator=='ate', estimator:='ATE']
rdfm[estimator=='constrained', estimator:='CR-MCM']
rdfm[estimator=='full', estimator:='fully observational']
rdfm[, estimator:=factor(estimator)]
rdfm <- rdfm[gamma_ut_sign==1.]
rdfm[metric%in%c('ksi0', 'ksi00', 'ksi01'), treatment:="treatment: 0"]
rdfm[metric%in%c('ksi1', 'ksi10', 'ksi11'), treatment:="treatment: 1"]
rdfm[metric%in%c('ksi00', 'ksi10'), x:=0L]
rdfm[metric%in%c('ksi01', 'ksi11'), x:=1L]
rdfm[, x:=factor(x)]

# make labels for plots
rdfm[, `:=`(
  oru=round(exp(au), 2),
  ortu=round(exp(atu), 2)
  )]
oru_levels = c(1,2,5)
oru_labels = c(
 expression(alpha[u]~'='~log(1)),
 expression(alpha[u]~'='~log(2)),
 expression(alpha[u]~'='~log(5))
)
ortu_labels = c(
 expression(alpha[tu]~'='~log(1)),
 expression(alpha[tu]~'='~log(2)),
 expression(alpha[tu]~'='~log(5))
)

rdfm[, oru_label:=factor(oru,
                         levels=oru_levels,
                         labels=oru_labels)]

rdfm[, ortu_label:=factor(ortu,
                         levels=oru_levels,
                         labels=ortu_labels)]


ggplot(rdfm[metric=='pehe'], aes(x=ax, y=value, col=estimator)) +
  geom_point() + geom_line() + 
  facet_grid(ortu_label~oru_label, labeller=label_parsed) +
  scale_color_manual(values=color_map) + 
  labs(y='PEHE', x=expression(alpha[x]))

width=20
# ratio=2.5
ratio = 1
height=width / ratio


ggsave(file.path(figdir, 'confounding_pehe.pdf'),
       width=width, height=height, units='cm', dpi=300)

rdfm[metric%in%c('ksi00', 'ksi01', 'ksi10', 'ksi11')] |>
  ggplot(aes(x=ax,y=value,col=estimator)) + 
  geom_point() + geom_line(aes(linetype=x)) +
  facet_grid(ortu_label+treatment~oru_label, labeller=label_parsed) +
  scale_color_manual(values=color_map) + 
  labs(y='y - y_hat', x=expression(alpha[x]))

ggsave(file.path(figdir, 'confounding_ksi.pdf'),
       width=width, height=height, units='cm', dpi=300)

rdfm[metric%in%c('ksi0', 'ksi1')] |>
  ggplot(aes(x=ax,y=value,col=estimator)) + 
  geom_point() + geom_line(aes(linetype=treatment)) +
  facet_grid(ortu_label~oru_label, labeller=label_parsed) +
  scale_color_manual(values=color_map) + 
  labs(y='E_x[(y - y_hat)^2]', x=expression(alpha[x]))

ggsave(file.path(figdir, 'confounding_ksi_mse.pdf'),
       width=width, height=height, units='cm', dpi=300)

  

