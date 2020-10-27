//STARTWHOLE
static char help[] = "Solve a mxm linear system using KSP.\n"
" use KSP 方法"
"注意到当m太大的时候，矩阵求解出现异常";

#include <petsc.h>

extern PetscErrorCode matrix(Mat , PetscInt);// form A
extern PetscErrorCode right(Vec ,PetscInt);//form b
extern PetscErrorCode exact(Vec ,PetscInt);//form xexact

int main(int argc,char **args) {
    PetscErrorCode ierr;
    Vec x,b,xexact;
    Mat A;
    KSP ksp;
    PetscInt m = 4;
    PetscReal err,errnorm;

    PetscInitialize(&argc,&args,NULL,help);

    ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"f_","options for f",""); CHKERRQ(ierr);
    ierr = PetscOptionsInt("-m","dimension of linear system","f.c",m,&m,NULL); CHKERRQ(ierr);//create an integer
    ierr = PetscOptionsEnd(); CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
    ierr = VecSetSizes(b,PETSC_DECIDE,m);CHKERRQ(ierr);
    ierr = VecSetFromOptions(b);CHKERRQ(ierr);
    ierr = right(b ,m);CHKERRQ(ierr);// set the value of b
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

    

    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m,m);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(A,"a_"); CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = matrix(A ,m);CHKERRQ(ierr);//set the value of A
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = VecDuplicate(b,&x);CHKERRQ(ierr);// make best of the location of b to store the x
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
/***
    ierr = PetscPrintf(PETSC_COMM_WORLD,
    "error for m = %d system , the numerical solution is \n",m); CHKERRQ(ierr);

    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);//这行命令会在执行过程中打印x
***/
//必须要把xexact这部分放到x下面
    ierr = VecDuplicate(x,&xexact); CHKERRQ(ierr);// make best of the location of b to store the xexact
    ierr = exact(xexact ,m);CHKERRQ(ierr);// set the value of xexact
    ierr = VecAssemblyBegin(xexact);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(xexact);CHKERRQ(ierr);
/***
    ierr = PetscPrintf(PETSC_COMM_WORLD,
    "error for m = %d system , the exact solution is \n",m); CHKERRQ(ierr);

    ierr = VecView(xexact,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
***/
    ierr = VecAXPY(x,-1.0,xexact); CHKERRQ(ierr);
    ierr = VecNorm(x,NORM_2,&errnorm); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
    "error for m = %d system is |x-xexact|_2 = %.1e\n",m,errnorm); CHKERRQ(ierr);

    ierr = VecAXPY(xexact,0.0,x); CHKERRQ(ierr);
    ierr = VecNorm(xexact,NORM_2,&err); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,
    "error for m = %d system is |x-xexact|_2/|xexact|_2 = %.1e\n",m,errnorm/err); CHKERRQ(ierr);



    KSPDestroy(&ksp);MatDestroy(&A);
    VecDestroy(&x);VecDestroy(&b);//VecDestroy(&xexact);
    return PetscFinalize();
}
//ENDWHOLE
PetscErrorCode matrix(Mat A , PetscInt m) {
    PetscErrorCode ierr;
    PetscInt i,j[4];
    PetscReal v[4];

    for(i = 0; i < m;i++) {
	if (i == 0) {
	    j[0] = 0,j[1] = 1;
	    v[0] = 6.0,v[1] = 1.0;
	    ierr = MatSetValues(A,1,&i,2,j,v,INSERT_VALUES);CHKERRQ(ierr);
	}
	else {
	    j[0] = i - 1,j[1] = i,j[2] = i + 1;
	    v[0] = 8.0,v[1] = 6.0,v[2] = 1.0;
	    if (i == m - 1) {
		ierr = MatSetValues(A,1,&i,2,j,v,INSERT_VALUES);CHKERRQ(ierr);
	    }
	    else {
		ierr = MatSetValues(A,1,&i,3,j,v,INSERT_VALUES);CHKERRQ(ierr);
	    }
	}
    }
    
    return 0;
}
PetscErrorCode right(Vec b , PetscInt m) {
    PetscErrorCode ierr;
    PetscInt i;
    PetscReal *ab;
 
    ierr = VecGetArray(b,&ab);CHKERRQ(ierr);
    for (i = 0; i < m; i++) {
	if (i == 0) {
	    ab[i] = 7.0;
	}
	else {
	    if (i == m - 1) {
		ab[i] = 14.0;
	    }
	    else {
	        ab[i] = 15.0;
	    }
	}
    }
    ierr = VecRestoreArray(b,&ab);CHKERRQ(ierr);
    return 0;
}

PetscErrorCode exact(Vec xexact , PetscInt m) {
    PetscErrorCode ierr;
    PetscInt i;
    PetscReal *ab;
 
    ierr = VecGetArray(xexact,&ab);CHKERRQ(ierr);
    for (i = 0; i < m; i++) {
	if (i == 0) {
	    ab[i] = 1.0;
	}
	else {
	    if (i == m - 1) {
		ab[i] = 1.0;
	    }
	    else {
	        ab[i] = 1.0;
	    }
	}
    }
    ierr = VecRestoreArray(xexact,&ab);CHKERRQ(ierr);
    return 0;
}

