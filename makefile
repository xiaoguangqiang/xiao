include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
CFLAGS += -pedantic -std=c99
#f和fu都是使用KSP方法求解线性系统

tri: tri.o
	-${CLINKER} -o tri tri.o  ${PETSC_LIB}
	${RM} tri.o
fun: fun.o
	-${CLINKER} -o fun fun.o  ${PETSC_LIB}
	${RM} fun.o
fu: fu.o
	-${CLINKER} -o fu fu.o  ${PETSC_LIB}
	${RM} fu.o
f: f.o
	-${CLINKER} -o f f.o  ${PETSC_LIB}
	${RM} f.o



runtri_1:
	-@../testit.sh tri "-a_mat_view ::ascii_dense" 1 1

runtri_2:
	-@../testit.sh tri "-tri_m 1000 -ksp_rtol 1.0e-4 -ksp_type cg -pc_type bjacobi -sub_pc_type jacobi -ksp_converged_reason" 2 2
runfun_1:
	-@../testit.sh fun "-a_mat_view ::ascii_dense" 1 1

runfun_2:
	-@../testit.sh fun "-fun_m 1000 -ksp_rtol 1.0e-4 -ksp_type cg -pc_type bjacobi -sub_pc_type jacobi -ksp_converged_reason" 2 2

runfu_1:
	-@../testit.sh fu "-a_mat_view ::ascii_dense" 1 1

runfu_2:
	-@../testit.sh fu "-fun_m 1000 -ksp_rtol 1.0e-10 -ksp_type cg -pc_type bjacobi -sub_pc_type jacobi -ksp_converged_reason" 2 2
runf_1:
	-@../testit.sh f "-a_mat_view ::ascii_dense" 1 1

runf_2:
	-@../testit.sh f "-fun_m 1000 -ksp_rtol 1.0e-10 -ksp_type cg -pc_type bjacobi -sub_pc_type jacobi -ksp_converged_reason" 2 2



test_tri: runtri_1 runtri_2
test_fun: runfun_1 runfun_2
test_fu: runfu_1 runfu_2
test_f: runf_1 runf_2


test: test_tri test_fun test_fu test_f

# etc

.PHONY: distclean runtri_1 runtri_2 runfun_1 runfun_2 runfu_1 runfu_2 runf_1 runf_2 test test_tri test_fun test_fu test_f

distclean:
	@rm -f *~ tri fun fu f *tmp
	 

