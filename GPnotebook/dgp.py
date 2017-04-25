# DGP Implementation borrowing routines from GPflow
import tensorflow as tf
import numpy as np
from . model import GPModel
from . param import Param , DataHolder
from . mean_functions import Zero
from . import likelihoods
from . tf_hacks import eye
from . import transforms
from . import kernels as ke
from . import kernel_expectations as ks
class SGPR ( GPModel ) :
    def __init__ (self , X, Y, mean_function = Zero ()):
        self .D = X. shape []
        self .N = X. shape []
        self .M =X= DataHolder (X, on_shape_change =’pass ’)
        Y= DataHolder (Y, on_shape_change =’pass ’)
        self .X=X; self .Y=Y;
        self . kern= ke. RBF ( self .D)
        self . likelihood= likelihoods . Gaussian ()
        self . likelihood= likelihoods . Gaussian ()
        GPModel . __init__ (self , X, Y, self .kern, self . likelihood, mean_function )
        self . kern= ke. RBF ( self .D)
        ND=np. ones (( self .N, self .D) )
        MD = np. reshape (np. linspace (, , ( self .M* self .D)),( self .M, self .D) )
        self .X= Param (Y /)
        self . X_var = Param (ND , transforms . positive )
        self .Z= Param (MD)
        self .Z = Param (MD)
        def build_likelihood_gplvm ( self ) :
            """
            Construct a tensorflow function to compute the bound on the marginal
            likelihood .
            """
            Y = self .Y;
            Z = self .Z;
            K = self . kern;
            X = self .XX_var = self . X_var ;
            num_inducing = tf. shape (Z)[]
            psi, psi, psi= ks. build_psi_stats (Z, K, X, X_var )
            self . psi= psiself . psi= psiself . psi= psiKuu = K.K(Z) + eye( num_inducing )*e -L = tf. cholesky (Kuu)
            sigma= self . likelihood. variance
            sigma = tf. sqrt ( sigma)
            # Compute intermediate matrices
            A = tf. matrix_triangular_solve (L, tf. transpose ( psi) , lower = True ) / sigma
            tmp = tf. matrix_triangular_solve (L, psi, lower = True )
            AAT = tf. matrix_triangular_solve (L, tf. transpose ( tmp ) , lower = True ) / sigmaB = AAT + eye ( num_inducing )
            LB = tf. cholesky (B)
            log_det_B = . * tf. reduce_sum (tf.log(tf. diag_part (LB)))
            c = tf. matrix_triangular_solve (LB , tf. matmul (A, Y) , lower = True ) / sigma
            # KL[q(x) || p(x) ]
            NQ = tf. cast (tf. size (X) , tf. float)
            D = tf. cast (tf. shape (Y)[], tf. float)
            KL = -.* tf. reduce_sum (tf. log ( X_var ) )
            # compute log marginal bound
            ND = tf. cast (tf. size (Y) , tf. float)
            bound = -.* ND * tf. log (* np.pi * sigma)
            bound += -.* D * log_det_B
            bound += -.* tf. reduce_sum (tf. square (Y)) / sigmabound += .* tf. reduce_sum (tf. square (c) )
            bound += -.* D * (tf. reduce_sum ( psi) / sigma-
            tf. reduce_sum (tf. diag_part (AAT)))
            bound -= KL
            return bound
            def build_likelihood_SGPR ( self ) :
                """
                Construct a tensorflow function to compute the bound on the marginal
                likelihood . For a derivation of the terms in here , see the associated
                SGPR notebook .
                """
                Y_var = self . X_var
                Y= self .X; Z= self .Z; K= self . kern; X= self .X
                num_inducing = tf. shape (Z)[]
                num_data = tf. cast (tf. shape (Y)[], tf. float)
                output_dim = tf. cast (tf. shape (Y)[], tf. float)
                Kdiag =K. Kdiag (X)
                Kuf = K.K(Z, X)
                Kuu = K.K(Z) + eye( num_inducing )*e -L = tf. cholesky (Kuu)
                sigma = tf. sqrt ( self . likelihood. variance )
                # Compute intermediate matrices
                A = tf. matrix_triangular_solve (L, Kuf , lower = True ) / sigma
                AAT = tf. matmul (A, tf. transpose (A) )
                B = AAT + eye ( num_inducing )
                LB = tf. cholesky (B)
                Aerr = tf. matmul (A, Y)
                c = tf. matrix_triangular_solve (LB , Aerr , lower = True ) / sigma
                # compute log marginal bound
                bound = -.* num_data * output_dim * np. log (* np.pi)
                bound += - output_dim * tf. reduce_sum (tf. log (tf. diag_part (LB)))
                bound -= .* num_data * output_dim * tf. log ( self . likelihood. variance )
                bound += -.* tf. reduce_sum (tf. square (Y))/ self . likelihood. variance
                bound += .* tf. reduce_sum (tf. square (c) )
                bound += -.* tf. reduce_sum ( Kdiag ) / self . likelihood. variance
                bound += .* tf. reduce_sum (tf. diag_part ( AAT ) )
                bound -= .* tf. reduce_sum ( Y_var ) / self . likelihood. variance
                return bound
                def build_likelihood ( self ) :
                    bound = self . build_likelihood_SGPR ()
                    bound += self . build_likelihood_gplvm ()
                    return bound
                    def pred (self , X, u, mu ,K) :
                        Kmm = K.K(u, u) + eye( self .M)*e -Knm = K.K(X, u)
                        posterior_mean = tf. matmul (Knm , tf. matrix_solve (Kmm , mu) )
                        Knn = K.K(X, X)
                        full_cov = Knn - tf. matmul (Knm , tf. matrix_solve (Kmm , tf. transpose ( Knm )))
                        return posterior_mean , full_cov
                        def build_predict ( self ) :
                            Y = self .Y;
                            Z = self .Z;
                            K = self . kern;
                            X = self .XX_var = self . X_var ;
                            Kmm =K.K(Z,Z) + eye( self .M)*e -Kmn = K.K(Z, X)
                            psi, psi, Kmnnm = ks. build_psi_stats (Z, K, X, X_var )
                            sigma= self . likelihood. variance
                            A_I = Kmnnm / sigma+ Kmm
                            Sig = tf. matmul (Kmm , tf. matrix_solve (A_I , Kmm ) )
                            mu = tf. matmul (Kmm , tf. matrix_solve (A_I , tf. matmul (Kmn , Y)))/ sigmareturn mu , Sig
                            def build_other ( self ) :
                                mu , Sig = self . build_predict ()
                                mean , cov= self . pred ( self .X, self .Z,mu , self . kern)
                                return mean , tf. diag_part ( cov )
