package org.hucompute.textimager.reader;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.apache.uima.UimaContext;
import org.apache.uima.collection.CollectionException;
import org.apache.uima.fit.descriptor.ConfigurationParameter;
import org.apache.uima.fit.util.JCasUtil;
import org.apache.uima.jcas.JCas;
import org.apache.uima.resource.ResourceInitializationException;
import org.hucompute.services.type.biocreative.Abstract;
import org.hucompute.services.type.biocreative.Title;

import de.tudarmstadt.ukp.dkpro.core.api.io.JCasResourceCollectionReader_ImplBase;
import de.tudarmstadt.ukp.dkpro.core.api.metadata.type.DocumentMetaData;
import de.tudarmstadt.ukp.dkpro.core.api.ner.type.NamedEntity;

public class BioCreativeReader extends JCasResourceCollectionReader_ImplBase {

	/**
	 * Write token annotations to the CAS.
	 */
	public static final String PARAM_ANNOTATION_FILE = "PARAM_ANNOTATION_FILE";
	@ConfigurationParameter(name = PARAM_ANNOTATION_FILE, mandatory = false)
	private String annotationFile;
	

	/**
	 * Write token annotations to the CAS.
	 */
	public static final String PARAM_FILTER_FILES_ONLY_WITH_ANNOTATIONS = "PARAM_FILTER_FILES_ONLY_WITH_ANNOTATIONS";
	@ConfigurationParameter(name = PARAM_FILTER_FILES_ONLY_WITH_ANNOTATIONS, mandatory = true)
	private boolean filter;

	private Iterator<String>inputLines;
	private int counter = 0;
	
	private HashMap<String, ArrayList<String>>annotations;
	@Override
	public void initialize(UimaContext arg0) throws ResourceInitializationException {
		super.initialize(arg0);
		if(annotationFile!=null){
			annotations = new HashMap<>();
			try {
				List<String>lines = FileUtils.readLines(new File(annotationFile));
				for (String string : lines) {
					String id = string.split("\t")[0];
					if(!annotations.containsKey(id))
						annotations.put(id, new ArrayList<>());
					annotations.get(id).add(string);
				}
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			System.out.println(annotations.size());
		}
		
		int countAnnotations = 0;
		for (ArrayList<String> annotations : this.annotations.values()) {
			countAnnotations+=annotations.size();
		}
		System.out.println(countAnnotations);
		
		Resource file = nextFile();
		try {
			inputLines = FileUtils.readLines(new File(file.getLocation().replace("file:", ""))).iterator();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	String string;
	String id; 	
	@Override
	public boolean hasNext() throws IOException, CollectionException {
		if(!inputLines.hasNext())
			return false;
		// TODO Auto-generated method stub
		do{
			if(!inputLines.hasNext())
				return false;
			string = inputLines.next();
			id = string.split("\t")[0];
		}
		while(!annotations.containsKey(id) && filter);
		return true;// && counter < 10;
	}
	

	@Override
	public void getNext(JCas aJCas) throws IOException, CollectionException {
		String inputtext = "";
//		String string = inputLines.next();
//		String id = string.split("\t")[0];

//		while(!annotations.containsKey(id) && filter){
//			if(!inputLines.hasNext())
//				return;
//			string = inputLines.next();
//			id = string.split("\t")[0];
//		}
		
		String title = string.split("\t")[1];
		String abstrakt = string.split("\t")[2];
		inputtext+=title + ".\t";
		
		DocumentMetaData meta = DocumentMetaData.create(aJCas);
		meta.setDocumentId(id);

		Title titleTag = new Title(aJCas, 0, title.length()+1);
		titleTag.setId(id);
		titleTag.addToIndexes();

		Abstract abstractTag = new Abstract(aJCas, inputtext.length(), inputtext.length()+ abstrakt.length());
		abstractTag.setId(id);
		abstractTag.addToIndexes();
		inputtext += abstrakt;
		aJCas.setDocumentText(inputtext);
		aJCas.setDocumentLanguage(getLanguage());
		
		if(annotations.containsKey(id)){
			ArrayList<String> listannotations = annotations.get(id);
			for (String annotation : listannotations) {
				BioAnnotation anno = new BioAnnotation(annotation);
				if(anno.typ.equals("T")){
					NamedEntity ne = new NamedEntity(aJCas, anno.begin, anno.end);
					ne.setValue(anno.klasse);
					ne.addToIndexes();
				}
				if(anno.typ.equals("A")){
					NamedEntity ne = new NamedEntity(aJCas, anno.begin+abstractTag.getBegin(), anno.end+abstractTag.getBegin());
					ne.setValue(anno.klasse);
					ne.addToIndexes();
				}
			}
			if(annotations.get(id).size() != JCasUtil.select(aJCas, NamedEntity.class).size()){
				System.out.println(id);
			}
		}
		counter++;		
	}
	
	private static class BioAnnotation{
		String id;
		String typ;
		int begin;
		int end;
		String name;
		String klasse;
		public BioAnnotation(String input){
			String split[] = input.split("\t");
			id = split[0];
			typ = split[1];
			begin = Integer.parseInt(split[2]);
			end = Integer.parseInt(split[3]);
			name = split[4];
			klasse = split[5];
		}

		@Override
		public String toString() {
			return id+"\t"+typ+"\t"+begin+"\t"+end+"\t"+name+"\t"+klasse;
		}
	}

}
