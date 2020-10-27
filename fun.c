//STARTWHOLE
static char help[] = "Solve a 4x4 linear system using KSP.\n";

#include <petsc.h>

extern PetscErrorCode matrix(Mat);// form A
extern PetscErrorCode right(Vec);//form b

int main(int argc,char **args) {
    PetscErrorCode ierr;
    Vec x,b;
    Mat A;
    KSP ksp;
    

    PetscInitialize(&argc,&args,NULL,help);

    ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
    ierr = VecSetSizes(b,PETSC_DECIDE,4);CHKERRQ(ierr);
    ierr = VecSetFromOptions(b);CHKERRQ(ierr);
    ierr = right(b);CHKERRQ(ierr);// set the value of A
    ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,4,4);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = matrix(A);CHKERRQ(ierr);//set the value of A
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = VecDuplicate(b,&x);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
    ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


    KSPDestroy(&ksp);MatDestroy(&A);
    VecDestroy(&x);VecDestroy(&b);
    return PetscFinalize();
}
//ENDWHOLE
PetscErrorCode matrix(Mat A) {
    PetscErrorCode ierr;
    PetscInt i,j[4] = {0,1,2,3};
    PetscReal aA[4][4] = {{1.0,0.0,0.0,0.0},
		    	  {0.0,1.0,3.0,0.0},
		    	  {0.0,0.0,2.0,0.0},
		    	  {0.0,0.0,3.0,4.0}};
    for (i = 0; i < 4; i++) {
	ierr = MatSetValues(A,1,&i,4,j,aA[i],INSERT_VALUES);CHKERRQ(ierr);
    }
    
    return 0;
}
PetscErrorCode right(Vec b) {
    PetscErrorCode ierr;
    PetscInt i;
    PetscReal *ab;
 
    ierr = VecGetArray(b,&ab);CHKERRQ(ierr);
    for (i = 0; i < 4; i++) {
	ab[i] = i;
    }
    ierr = VecRestoreArray(b,&ab);CHKERRQ(ierr);
    return 0;
}

