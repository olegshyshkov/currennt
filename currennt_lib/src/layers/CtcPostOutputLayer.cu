#include "CtcPostOutputLayer.hpp"
#include "../helpers/getRawPointer.cuh"
#include "../helpers/NumericLimits.cuh"
#include "../helpers/safeExp.cuh"
#include <iostream>

#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <iostream>

class LogScale {
    public:
        __host__ __device__ inline static real_t sum(real_t a, real_t b) {
            if (a < b) {
                real_t c = a; a = b; b = c;
            }
            real_t e = b - a;
            if (e > -helpers::NumericLimits<real_t>::expLimit()) {
                a += std::log(1 + exp(e));
            }
            return a;
        }
        __host__ __device__ inline static real_t sub(real_t a, real_t b) {
            if (a < b) {
                return helpers::NumericLimits<real_t>::logZero();
            }


            real_t e = b - a;
            if (e > -helpers::NumericLimits<real_t>::expLimit()) {
                a += log(1 - exp(e));
            }


            return a;
        }


        __host__ __device__ inline static real_t mult(real_t a, real_t b) {
            return a + b;
        }


        __host__ __device__ inline static real_t div(real_t a, real_t b) {
            return a - b;
        }


        __host__ __device__ inline static real_t linear_log(real_t a) {
            return a;
        }


        struct LinearLog {
            __host__ __device__ real_t operator()(const real_t& a) const { return a; }
        };


        __host__ __device__ inline static real_t to_scale(real_t a) {
            if (a <= helpers::NumericLimits<real_t>::min()) {
                return helpers::NumericLimits<real_t>::logZero();
            }
            else {
                return log(a);
            }
        }


        __host__ __device__ inline static real_t to_linear(real_t a) {
            return helpers::safeExp(a);
        }


        __host__ __device__ inline static real_t one() {
            return 0;
        }


        __host__ __device__ inline static real_t zero() {
            return helpers::NumericLimits<real_t>::logZero();
        }
};


typedef LogScale sc;



namespace internal {
    namespace {
        struct ComputeAlphaFn
        {
            public:
                const real_t *prevAlphas;
                const real_t *inputs;
                const real_t *setAlpha;


                const char *patTypes;


                const int *fmap;
                const int *imap;


                int maxWordLen;
                int parallelSequences;


                __host__ __device__ real_t operator() (const int &outputIdx)
                {
                    int patIdx = outputIdx / maxWordLen;
                    int blockIdx = outputIdx % (maxWordLen * parallelSequences);


                    if (patTypes[patIdx] == PATTYPE_NONE) {
                        return sc::zero();
                    }


                    if (patTypes[patIdx] == PATTYPE_FIRST) {
                        return sc::mult(setAlpha[blockIdx], sc::to_scale(inputs[imap[blockIdx]]));
                    }


                    real_t output = sc::zero();


                    for (int j = fmap[blockIdx]; j <= blockIdx; ++j) {
                        output = sc::sum(output, prevAlphas[j]);
                    }


                    output = sc::mult(output, sc::to_scale(inputs[imap[blockIdx]]));


                    return output;
                }
        };


        struct ComputeBetaFn
        {
            public:
                const real_t *nextBeta;
                const real_t *inputs;
                const real_t *setBeta;
                const char *patTypes;


                const int *gmap;
                const int *imap;


                int maxWordLen;
                int parallelSequences;


                __host__ __device__ real_t operator() (const int &outputIdx)
                {
                    int patIdx = outputIdx / maxWordLen;
                    int blockIdx = outputIdx % (maxWordLen * parallelSequences);


                    if (patTypes[patIdx] == PATTYPE_LAST) {
                        return setBeta[blockIdx];
                    }


                    if (patTypes[patIdx] == PATTYPE_NONE) {
                        return sc::zero();
                    }


                    real_t output = sc::zero();


                    for (int j = blockIdx; j <= gmap[blockIdx]; ++j) {
                        real_t s = sc::mult(nextBeta[j], sc::to_scale(inputs[imap[j]]));
                        output = sc::sum(output, s);
                    }


                    return output;
                }
        };


        struct ComputeWordErrorFn
        {
            const real_t *alphas;
            const real_t *betas;
            int maxWordLen;


            public:
            __host__ __device__ real_t operator() (const int &outputIdx)
            {
                int i = outputIdx * maxWordLen;
                return sc::sum(sc::mult(betas[i], alphas[i]), sc::mult(betas[i + 1], alphas[i + 1]));
            }
        };


        struct ComputeOutputErrorPhaseFn
        {
            const real_t *alphas;
            const real_t *betas;
            const real_t *wordErrors;
            const char *patTypes;


            const int *imap;
            int maxWordLen;
            int parallelSequences;
            int prevLayerSize;


            public:
            __host__ __device__ real_t operator() (const real_t &y, const int &outputIdx)
            {
                int seqIdx = (outputIdx / prevLayerSize) % parallelSequences;
                int blockIdx = outputIdx % (prevLayerSize * parallelSequences);
                int patIdx = outputIdx / prevLayerSize;
                if (patTypes[patIdx] == PATTYPE_NONE) {
                    return 0.f;
                }
                real_t output = sc::zero();
                real_t num = sc::zero();
                real_t den = sc::zero();


                for (int i = 0; i < maxWordLen; ++i) {
                    real_t cur = sc::mult(alphas[patIdx * maxWordLen + i], betas[patIdx * maxWordLen + i]);
                    if (imap[seqIdx * maxWordLen + i] == blockIdx) {
                        num = sc::sum(num, cur);
                    }
                    den = sc::sum(den, cur);
                }
                real_t error = -sc::to_linear(sc::div(num, den)) / y;
                return error;
            }
        };
    }
}


namespace layers {


    template <typename TDevice>
        CtcPostOutputLayer<TDevice>::CtcPostOutputLayer(const helpers::JsonValue &layerChild, Layer<TDevice> &precedingLayer)
        : PostOutputLayer<TDevice>(layerChild, precedingLayer, 1), prevLayerSize(precedingLayer.size())
        {
        }


    template <typename TDevice>
        CtcPostOutputLayer<TDevice>::~CtcPostOutputLayer()
        {
        }


    template <typename TDevice>
        const std::string& CtcPostOutputLayer<TDevice>::type() const
        {
            static const std::string s("ctc");
            return s;
        }

    template <typename TDevice>
        void CtcPostOutputLayer<TDevice>::findAlphas()
        {
            m_alphas.resize(maxWordLen * this->parallelSequences() * this->curMaxSeqLength());


            real_vector &y = this->_actualOutputs();


            internal::ComputeAlphaFn fn;
            fn.prevAlphas = helpers::getRawPointer(m_alphas);
            fn.fmap = helpers::getRawPointer(fmap);
            fn.imap = helpers::getRawPointer(imap);
            fn.inputs = helpers::getRawPointer(this->_actualOutputs());
            fn.patTypes = helpers::getRawPointer(this->patTypes());
            fn.setAlpha = helpers::getRawPointer(this->setAlpha);
            fn.maxWordLen = this->maxWordLen;
            fn.parallelSequences = this->parallelSequences();


            int alphaTimeSize = this->maxWordLen * this->parallelSequences();
            int inputTimeSize = this->prevLayerSize * this->parallelSequences();


            for (int t = 0; t < this->curMaxSeqLength(); ++t) {


                thrust::transform(
                        thrust::counting_iterator<int>(t*alphaTimeSize),
                        thrust::counting_iterator<int>(t*alphaTimeSize) + alphaTimeSize,
                        m_alphas.begin() + t*alphaTimeSize,
                        fn
                        );


                fn.inputs += inputTimeSize;
                if(t != 0) fn.prevAlphas += alphaTimeSize;
            }
        }


    template <typename TDevice>
        void CtcPostOutputLayer<TDevice>::findBetas()
        {
            m_betas.resize(maxWordLen * this->parallelSequences() * this->curMaxSeqLength());


            internal::ComputeBetaFn fn;
            fn.nextBeta = helpers::getRawPointer(m_betas) + m_betas.size();
            fn.gmap = helpers::getRawPointer(gmap);
            fn.imap = helpers::getRawPointer(imap);
            fn.inputs = helpers::getRawPointer(this->_actualOutputs()) + this->prevLayerSize * this->parallelSequences() * this->curMaxSeqLength();
            fn.patTypes = helpers::getRawPointer(this->patTypes());
            fn.setBeta = helpers::getRawPointer(this->setBeta);
            fn.maxWordLen = this->maxWordLen;
            fn.parallelSequences = this->parallelSequences();


            int betaTimeSize = maxWordLen * this->parallelSequences();
            int inputTimeSize = prevLayerSize * this->parallelSequences();


            for (int t = this->curMaxSeqLength() - 1; t >= 0; --t) {
                thrust::transform(
                        thrust::counting_iterator<int>(t*betaTimeSize),
                        thrust::counting_iterator<int>(t*betaTimeSize) + betaTimeSize,
                        m_betas.begin() + t*betaTimeSize,
                        fn
                        );


                fn.inputs -= inputTimeSize;
                fn.nextBeta -= betaTimeSize;


            }
        }

    template <typename TDevice>
        real_t CtcPostOutputLayer<TDevice>::calculateError()
        {
            wordErrors.resize(this->curNumSeqs());


            internal::ComputeWordErrorFn fn;
            fn.alphas = helpers::getRawPointer(m_alphas);
            fn.betas = helpers::getRawPointer(m_betas);
            fn.maxWordLen = this->maxWordLen;


            thrust::transform(
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(this->curNumSeqs()),
                    wordErrors.begin(),
                    fn
                    );
            real_t error = -(thrust::transform_reduce(wordErrors.begin(), wordErrors.end(), sc::LinearLog(), 0.f, thrust::plus<real_t>()));
            return error;
        }


    template <typename TDevice>
        void CtcPostOutputLayer<TDevice>::computeForwardPass()
        {
            findAlphas();
            findBetas();
        }


    template <typename TDevice>
        void CtcPostOutputLayer<TDevice>::computeBackwardPass()
        {
            real_vector &outputErrors = this->_outputErrors();
            real_vector &input = this->_actualOutputs();


            internal::ComputeOutputErrorPhaseFn fn;
            fn.alphas = helpers::getRawPointer(m_alphas);
            fn.betas = helpers::getRawPointer(m_betas);
            fn.wordErrors = helpers::getRawPointer(wordErrors);
            fn.imap = helpers::getRawPointer(imap);
            fn.patTypes = helpers::getRawPointer(this->patTypes());
            fn.maxWordLen = this->maxWordLen;
            fn.prevLayerSize = this->prevLayerSize;
            fn.parallelSequences = this->curNumSeqs();


            thrust::transform(input.begin(), input.end(), thrust::counting_iterator<int>(0), outputErrors.begin(), fn);
        }


    template <typename TDevice>
        void CtcPostOutputLayer<TDevice>::loadSequences(const data_sets::DataSetFraction &fraction)
        {
            PostOutputLayer<TDevice>::loadSequences(fraction);


            real_vector &o = this->_targets();
            int n = this->curMaxSeqLength();


            std::vector< std::vector<int> > words(this->curNumSeqs());


            for (size_t i = 0; i < this->curNumSeqs(); ++i) {
                for (int j = i; int(o[j] + 0.5) != 0; j += this->curNumSeqs()) {
                    words[i].push_back(int(o[j] + 0.5));
                }
            }


            this->maxWordLen = 0;
            for(size_t i = 0; i < words.size(); ++i) {
                this->maxWordLen = std::max(this->maxWordLen, 2 * words[i].size() + 1);
            }


            this->computationSize = this->curNumSeqs() * this->maxWordLen;


            imap.resize(this->computationSize);


            setBeta = real_vector(this->computationSize, sc::zero());
            setAlpha = real_vector(this->computationSize, sc::zero());


            for (int i = 0; i < this->curNumSeqs(); ++i) {
                int word_offset = i * this->maxWordLen;
                int input_offset = i * this->prevLayerSize;


                std::fill(imap.begin() + word_offset, imap.begin() + word_offset + maxWordLen, input_offset);


                for (size_t j = 0; j < words[i].size(); ++j) {
                    imap[word_offset + 2 * j + 1] += words[i][j];
                }


                setAlpha[word_offset] = sc::one();
                setAlpha[word_offset + 1] = sc::one();


                setBeta[word_offset + 2 * words[i].size()] = sc::one();
                setBeta[word_offset + 2 * words[i].size() - 1] = sc::one();
            }


            fmap.resize(this->maxWordLen * this->curNumSeqs(), this->maxWordLen * this->curNumSeqs());
            gmap.resize(this->maxWordLen * this->curNumSeqs(), 0);


            for (int i = 0; i < this->curNumSeqs(); ++i) {
                int longWordSize = 2 * words[i].size() + 1;
                for (int u = 0; u < longWordSize; ++u) {
                    int let = (u % 2 == 0) ? 0 : (u / 2);
                    if (u % 2 == 0 || (let >= 1 && words[i][let] == words[i][let - 1])) {
                        fmap[i * maxWordLen + u] = std::max(0, u - 1) + i * maxWordLen;
                    }
                    else {
                        fmap[i * maxWordLen + u] = std::max(0, u - 2) + i * maxWordLen;
                    }


                    if (u % 2 == 0 || ((let + 1 < words[i].size()) && (words[i][let + 1] == words[i][let]))) {
                        gmap[i * maxWordLen + u] = std::min(longWordSize - 1, u + 1) + i * maxWordLen;
                    }
                    else {
                        gmap[i * maxWordLen + u] = std::min(longWordSize - 1, u + 2) + i * maxWordLen;
                    }
                }
            }

        }


    template class CtcPostOutputLayer<Cpu>;
    template class CtcPostOutputLayer<Gpu>;
};
