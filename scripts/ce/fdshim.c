#define _GNU_SOURCE
#include <sys/types.h>
#include <sys/resource.h>
#include <dlfcn.h>
int getrlimit(__rlimit_resource_t r, struct rlimit *l){
  static int (*real)(__rlimit_resource_t, struct rlimit*)=0;
  if(!real) real=dlsym(RTLD_NEXT,"getrlimit");
  int rc=real(r,l);
  if(rc==0 && r==RLIMIT_NOFILE){ l->rlim_cur=1000000; l->rlim_max=1000000; }
  return rc;
}
int getrlimit64(__rlimit_resource_t r, struct rlimit64 *l){
  static int (*real)(__rlimit_resource_t, struct rlimit64*)=0;
  if(!real) real=dlsym(RTLD_NEXT,"getrlimit64");
  int rc=real(r,l);
  if(rc==0 && r==RLIMIT_NOFILE){ l->rlim_cur=1000000; l->rlim_max=1000000; }
  return rc;
}
int prlimit(pid_t p, __rlimit_resource_t r, const struct rlimit *n, struct rlimit *o){
  static int (*real)(pid_t,__rlimit_resource_t,const struct rlimit*,struct rlimit*)=0;
  if(!real) real=dlsym(RTLD_NEXT,"prlimit");
  int rc=real(p,r,n,o);
  if(rc==0 && o && r==RLIMIT_NOFILE){ o->rlim_cur=1000000; o->rlim_max=1000000; }
  return rc;
}
